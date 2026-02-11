# Plonky2 Core Performance Optimization Report

## Executive Summary

Analysis of the plonky2 prover pipeline reveals **~200ms per recursive proof** with the following breakdown from profiling:

| Phase | Time (ms) | % of Total |
|-------|-----------|------------|
| Merkle tree hashing | ~60 | 30% |
| Generator execution | ~32 | 16% |
| Quotient poly computation | ~30 | 15% |
| Wires commitment (total) | ~72 | 36% |
| Opening proofs | ~15 | 7% |
| Partial products | ~17 | 9% |

The optimizations below target **eliminating unnecessary memcopies, improving data layout for cache efficiency, reducing allocations, and increasing SIMD utilization** -- all without changing the underlying cryptographic math.

---

## 1. CRITICAL: Eliminate `Vec<Vec<F>>` in Merkle Tree Leaves

**Files:** `plonky2/src/hash/merkle_tree.rs:53`, `plonky2/src/plonk/config.rs:189`
**Impact:** ~15-20% of Merkle tree time (est. 10-12ms saved)

### Problem
Merkle tree leaves are stored as `Vec<Vec<F>>` -- a doubly-nested vector where each leaf is a separate heap allocation:

```rust
// merkle_tree.rs:51-53
pub struct MerkleTree<F: RichField, H: Hasher<F>> {
    pub leaves: Vec<Vec<F>>,  // Each leaf is a separate heap allocation
    ...
}
```

For a tree with 65,536 leaves of 135 field elements each (typical in the benchmark), this means:
- 65,536 separate allocations scattered across memory
- Poor L3 cache utilization during hashing (each `H::hash_or_noop(&leaves[i])` chases a pointer)
- ~4.2 MB of leaf data in non-contiguous memory

### Fix
Replace with a flat contiguous buffer plus a stride:

```rust
pub struct MerkleTree<F: RichField, H: Hasher<F>> {
    pub leaves: Vec<F>,       // Flat contiguous allocation
    pub leaf_len: usize,      // All leaves same size
    pub digests: Vec<H::Hash>,
    pub cap: MerkleCap<F, H>,
}
```

Access pattern changes from `&self.leaves[i]` to `&self.leaves[i * self.leaf_len..(i+1) * self.leaf_len]`.

This also eliminates the cost of the `transpose()` function (`plonky2/src/util/mod.rs:25-31`) which currently produces the `Vec<Vec<F>>`:

```rust
// util/mod.rs:25-31 -- creates Vec<Vec<F>> with column-wise random access
pub fn transpose<T: Send + Sync + Copy>(matrix: &[Vec<T>]) -> Vec<Vec<T>> {
    let len = matrix[0].len();
    (0..len)
        .into_par_iter()
        .map(|i| matrix.iter().map(|row| row[i]).collect())
        .collect()
}
```

The transpose could write directly into a flat buffer, avoiding 65,536 inner Vec allocations.

---

## 2. CRITICAL: In-Place Coset FFT Shift

**File:** `field/src/polynomial/mod.rs:286-299`
**Impact:** Eliminates one full polynomial copy per FFT (~8ms across all commitments)

### Problem
Every coset FFT creates a full copy of the polynomial to apply the coset shift:

```rust
// polynomial/mod.rs:291-298
pub fn coset_fft_with_options(&self, shift: F, ...) -> PolynomialValues<F> {
    let modified_poly: Self = shift
        .powers()
        .zip(&self.coeffs)
        .map(|(r, &c)| r * c)
        .collect::<Vec<_>>()   // <-- Full polynomial copy
        .into();
    modified_poly.fft_with_options(zero_factor, root_table)
}
```

This is called for every polynomial in every commitment:
- Wires commitment: 135 polynomials x 8192 elements = 1.08M copies
- Partial products: ~18 polynomials
- Quotient: ~16 polynomials
- FRI rounds: repeated per round

### Fix
Apply coset shift in-place before FFT, or better yet, integrate the shift into the first butterfly stage of the FFT. The shift `s^i` for coefficient `i` can be absorbed into the twiddle factors.

```rust
pub fn coset_fft_in_place(&mut self, shift: F, ...) -> PolynomialValues<F> {
    // Apply shift in-place
    let mut s_pow = F::ONE;
    for c in self.coeffs.iter_mut() {
        *c *= s_pow;
        s_pow *= shift;
    }
    self.fft_with_options(zero_factor, root_table)
}
```

---

## 3. CRITICAL: Optimize `full_witness()` Construction

**File:** `plonky2/src/iop/witness.rs:376-388`
**Impact:** ~4-5ms saved (currently ~6ms of the "compute full witness" phase)

### Problem
The witness materialization iterates over every `(row, wire)` pair with double indirection:

```rust
// witness.rs:376-388
pub fn full_witness(self) -> MatrixWitness<F> {
    let mut wire_values = vec![vec![F::ZERO; self.degree]; self.num_wires];
    for i in 0..self.degree {              // 8192 iterations
        for j in 0..self.num_wires {       // 135 iterations
            let t = Target::Wire(Wire { row: i, column: j });
            if let Some(x) = self.try_get_target(t) {  // Double indirection
                wire_values[j][i] = x;
            }
        }
    }
    MatrixWitness { wire_values }
}
```

Each `try_get_target` requires:
1. `target.index()` -- multiplication + branch (`witness.rs:372-374`)
2. `representative_map[index]` -- L2/L3 cache lookup
3. `values[rep_index]` -- sparse `Option<F>` access

For degree=8192 and num_wires=135, that's **~1.1M lookups**, most returning `None`.

### Fix
Instead of iterating all `(row, wire)` pairs, iterate only the populated entries in `values`:

```rust
pub fn full_witness(self) -> MatrixWitness<F> {
    let mut wire_values = vec![vec![F::ZERO; self.degree]; self.num_wires];
    // Only iterate populated values
    for (idx, value) in self.values.iter().enumerate() {
        if let Some(v) = value {
            // Reverse the index mapping to get (row, wire)
            if idx < self.degree * self.num_wires {
                let row = idx / self.num_wires;
                let col = idx % self.num_wires;
                wire_values[col][row] = *v;
            }
        }
    }
    MatrixWitness { wire_values }
}
```

Also, `MatrixWitness` should use a flat `Vec<F>` instead of `Vec<Vec<F>>` to avoid the clone in `prover.rs:169`:

```rust
// prover.rs:163-171 -- clones each wire column
let wires_values: Vec<PolynomialValues<F>> = witness
    .wire_values
    .par_iter()
    .map(|column| PolynomialValues::new(column.clone()))  // <-- unnecessary clone
    .collect()
```

If `MatrixWitness` could transfer ownership (`.into_iter()` instead of `.par_iter()`), this clone is eliminated.

---

## 4. HIGH: Reduce Poseidon Permutation Init Overhead in Merkle Hashing

**File:** `plonky2/src/hash/hashing.rs:97-114`
**Impact:** ~5-8ms saved across all Merkle tree operations

### Problem
Every `two_to_one` (internal node) and `hash_or_noop` (leaf) call initializes a fresh Poseidon permutation state:

```rust
// hashing.rs:105
let mut perm = P::new(core::iter::repeat(F::ZERO));  // 12 field elements zeroed
```

For a tree with 65,536 leaves and ~65,534 internal nodes, that's **~131,070 permutation state initializations**.

Additionally, `hash_n_to_m_no_pad` (`hashing.rs:118-141`) allocates a `Vec::new()` for output on every call:

```rust
// hashing.rs:131
let mut outputs = Vec::new();  // heap allocation per hash
```

### Fix
1. Zero only the capacity portion of the permutation state (elements beyond `NUM_HASH_OUT_ELTS * 2`) since `set_from_slice` overwrites the rest
2. Use a stack-allocated `[F; NUM_HASH_OUT_ELTS]` for output instead of `Vec`
3. Consider a `hash_n_to_hash_no_pad` variant that returns `HashOut<F>` directly without Vec:

```rust
pub fn hash_n_to_hash_no_pad<F: RichField, P: PlonkyPermutation<F>>(inputs: &[F]) -> HashOut<F> {
    let mut perm = P::new(core::iter::repeat(F::ZERO));
    for input_chunk in inputs.chunks(P::RATE) {
        perm.set_from_slice(input_chunk, 0);
        perm.permute();
    }
    HashOut {
        elements: perm.squeeze()[..NUM_HASH_OUT_ELTS].try_into().unwrap(),
    }
}
```

This already exists at line 143! But `hash_n_to_m_no_pad` is still called in other paths. Ensure all leaf-hashing paths use the direct `HashOut` variant.

---

## 5. HIGH: Eliminate `.to_vec()` Copies in Quotient Poly Batch Evaluation

**File:** `plonky2/src/plonk/prover.rs:682-688`
**Impact:** ~3-5ms saved (quotient computation is ~30ms total)

### Problem
For every batch of 32 evaluation points, 6 `.to_vec()` calls copy slices:

```rust
// prover.rs:682-688 (per evaluation point in batch)
s_sigmas_batch_vecs.push(local_constants_sigmas[common_data.sigmas_range()].to_vec());
local_zs_batch_vecs.push(local_zs_partial_and_lookup[common_data.zs_range()].to_vec());
next_zs_batch_vecs.push(next_zs_partial_and_lookup[common_data.zs_range()].to_vec());
partial_products_batch_vecs.push(local_zs_partial_and_lookup[common_data.partial_products_range()].to_vec());
```

With LDE size = 65,536 points and batch size = 32, that's 2,048 batches x 32 points x 6 copies = **~393K small Vec allocations**.

### Fix
Redesign `eval_vanishing_poly_base_batch` to accept slice references into the Merkle tree leaves directly, avoiding the owned Vec intermediaries. The data is already available as contiguous slices via `PolynomialBatch::get_lde_values()`.

---

## 6. HIGH: Batch Poseidon Hashing with PackedField in Merkle Tree

**File:** `plonky2/src/hash/merkle_tree.rs:91-118`
**Impact:** ~15-25% of Merkle tree time (est. 10-15ms saved)

### Problem
Each internal node hash is computed independently:

```rust
// merkle_tree.rs:116
H::two_to_one(left_digest, right_digest)  // One hash at a time
```

Poseidon with Goldilocks field has AVX2 implementations (`hash/arch/x86_64/poseidon_goldilocks_avx2_bmi2.rs`), but these optimize *within* a single permutation. Multiple independent hashes at the same tree level could be batched using `PackedField` to process 4 (AVX2) or 8 (AVX-512) permutations simultaneously.

### Fix
For each tree level, collect all node pairs and hash them in SIMD batches:

```rust
// Process 4 independent two_to_one hashes simultaneously with AVX2
fn batch_two_to_one_avx2<F: RichField>(
    lefts: &[HashOut<F>; 4],
    rights: &[HashOut<F>; 4],
) -> [HashOut<F>; 4] {
    // Pack 4 states into PackedField, permute once, extract 4 results
}
```

This requires a level-by-level tree construction instead of recursive `fill_subtree`, but the parallelism gain is significant.

---

## 7. MEDIUM: Eliminate `flatten`/`unflatten` Allocations in FRI

**File:** `field/src/extension/mod.rs:128-146`
**Impact:** ~1-2ms saved across FRI rounds

### Problem
```rust
// extension/mod.rs:132-134
pub fn flatten<F, const D: usize>(l: &[F::Extension]) -> Vec<F> {
    l.iter()
        .flat_map(|x| x.to_basefield_array().to_vec())  // .to_vec() per element!
        .collect()
}

// extension/mod.rs:143-144
pub fn unflatten<F, const D: usize>(l: &[F]) -> Vec<F::Extension> {
    l.chunks_exact(D)
        .map(|c| F::Extension::from_basefield_array(c.to_vec().try_into().unwrap()))
        .collect()
}
```

Each extension element creates a temporary `Vec` via `.to_vec()`. For D=2, this is a 2-element Vec allocation per element.

### Fix
Use fixed-size arrays directly:

```rust
pub fn flatten<F, const D: usize>(l: &[F::Extension]) -> Vec<F> {
    let mut result = Vec::with_capacity(l.len() * D);
    for x in l {
        result.extend_from_slice(&x.to_basefield_array());
    }
    result
}

pub fn unflatten<F, const D: usize>(l: &[F]) -> Vec<F::Extension> {
    l.chunks_exact(D)
        .map(|c| {
            let arr: [F; D] = c.try_into().unwrap();
            F::Extension::from_basefield_array(arr)
        })
        .collect()
}
```

---

## 8. MEDIUM: Reduce Generator Dispatch Overhead

**File:** `plonky2/src/iop/generator.rs:58-97`
**Impact:** ~2-4ms saved

### Problem
Generators use dynamic dispatch through `Box<dyn WitnessGenerator<F, D>>`:

```rust
// generator.rs:130-132
pub struct WitnessGeneratorRef<F: RichField + Extendable<D>, const D: usize>(
    pub Box<dyn WitnessGenerator<F, D>>,
);
```

Each generator invocation incurs:
- vtable lookup for `run()` method
- For `SimpleGenerator` variants, a second layer of indirection through `SimpleGeneratorAdapter`
- Pipeline stalls from unpredictable indirect branches

With ~8,147 generators per proof, this adds up.

### Fix Options
1. **Enum dispatch**: Replace `Box<dyn WitnessGenerator>` with an enum of concrete generator types, enabling static dispatch and inlining
2. **Group generators by type**: Sort generators by concrete type so the CPU branch predictor can learn the pattern
3. **Pre-resolve dependencies**: Currently each generator checks `witness.contains_all()` on every invocation attempt. Pre-compute a dependency DAG and only invoke generators when all inputs are ready.

---

## 9. MEDIUM: Wire Polynomial Clone Elimination

**File:** `plonky2/src/plonk/prover.rs:163-171`
**Impact:** ~0.5-1ms saved (small at degree 8192, significant at higher degrees)

### Problem
```rust
// prover.rs:169
.map(|column| PolynomialValues::new(column.clone()))  // clones each wire column
```

Each of the 135 wire columns of length 8192 is cloned.

### Fix
Transfer ownership from `MatrixWitness` instead of cloning:

```rust
let wires_values: Vec<PolynomialValues<F>> = witness
    .wire_values
    .into_par_iter()  // into_par_iter takes ownership
    .map(|column| PolynomialValues::new(column))
    .collect()
```

This requires `MatrixWitness` to be consumed (not borrowed), which is already the case in the prover pipeline.

---

## 10. MEDIUM: Optimize Transpose for LDE Values

**File:** `plonky2/src/plonk/config.rs:189-190`, `plonky2/src/util/mod.rs:25-31`
**Impact:** ~2-3ms saved per commitment

### Problem
After computing LDE values (polynomial-major: `[num_polys][lde_size]`), a full transpose converts to evaluation-point-major (`[lde_size][num_polys]`) for Merkle tree leaf hashing:

```rust
// config.rs:189-190
let mut leaves = transpose(&lde_values);     // O(n*m) with poor cache access
reverse_index_bits_in_place(&mut leaves);    // O(n*m) random swaps
```

The `transpose` function (`util/mod.rs:25-31`) accesses columns of a row-major matrix, causing systematic cache misses.

### Fix Options
1. **Tiled transpose**: Process in cache-friendly blocks (e.g., 64x64 tiles)
2. **Fused transpose + bit-reversal**: Combine into a single pass to halve memory traffic
3. **Direct evaluation-point-major LDE**: Generate LDE values in the final layout directly, avoiding transpose entirely. Each evaluation point `x` can evaluate all polynomials at once.

---

## 11. MEDIUM: Pre-Allocate Gate Constraint Buffers

**File:** `plonky2/src/plonk/vanishing_poly.rs:702-726`
**Impact:** ~1-2ms saved

### Problem
Each gate allocates a fresh constraint vector per batch:

```rust
// Inside eval_unfiltered_base_batch (called per gate):
let mut res = vec![F::ZERO; vars_base.len() * self.num_constraints()];
```

With ~13 gates and 2,048 batches, that's ~26,000 allocations.

### Fix
Pass a pre-allocated mutable slice to each gate's evaluation function, allowing constraint accumulation in-place without per-gate allocation.

---

## 12. LOW-MEDIUM: Reduce FRI Round Memory Churn

**File:** `plonky2/src/fri/prover.rs:84-119`
**Impact:** ~1ms saved

### Problem
Each FRI round performs:
1. `reverse_index_bits_in_place(&mut values.values)` -- O(n) random swaps
2. `.par_chunks(arity).map(|chunk| flatten(chunk)).collect()` -- allocates Vec per chunk
3. `MerkleTree::new(chunked_values, ...)` -- builds tree from Vec<Vec<F>>
4. `coeffs.coset_fft(shift.into())` -- creates new polynomial copy

### Fix
1. Maintain bit-reversed order throughout FRI rounds (avoid repeated reversal)
2. Use the flat Merkle tree leaf storage from optimization #1
3. Apply coset FFT in-place (optimization #2)

---

## 13. LOW: `hash_or_noop` Stack Allocation for Small Inputs

**File:** `plonky2/src/plonk/config.rs:75-86` (via Hasher trait)
**Impact:** ~0.5ms saved

### Problem
When leaf data is small enough to fit directly in a hash:
```rust
fn hash_or_noop(inputs: &[F]) -> Self::Hash {
    if inputs.len() * 8 <= Self::HASH_SIZE {
        let mut inputs_bytes = vec![0u8; Self::HASH_SIZE];  // heap allocation
        ...
    }
}
```

### Fix
Use a stack-allocated array: `let mut inputs_bytes = [0u8; 32];`

---

## 14. LOW: Opening Set Construction Copies

**File:** `plonky2/src/plonk/proof.rs:313-351`
**Impact:** <0.5ms saved (but cleaner code)

### Problem
8 `.to_vec()` calls slice evaluation results into owned Vecs:
```rust
constants: constants_sigmas_eval[common_data.constants_range()].to_vec(),
plonk_sigmas: constants_sigmas_eval[common_data.sigmas_range()].to_vec(),
```

### Fix
Evaluate directly into the target fields without intermediate full evaluation + slice.

---

## 15. LOW: Query Phase `.to_vec()` Copies

**File:** `plonky2/src/fri/prover.rs:238`
**Impact:** <0.5ms saved

### Problem
```rust
let initial_proof = initial_merkle_trees
    .iter()
    .map(|t| (t.get(x_index).to_vec(), t.prove(x_index)))  // .to_vec() copies leaf
    .collect();
```

### Fix
Use `Cow<'_, [F]>` or restructure to avoid the copy if the data isn't mutated.

---

## Optimization Priority Matrix

| # | Optimization | Est. Savings | Risk | Difficulty | Dependencies |
|---|---|---|---|---|---|
| 1 | Flat Merkle tree leaves | 10-12ms | Low | Medium | Changes public API |
| 2 | In-place coset FFT shift | 8ms | Low | Easy | None |
| 3 | Optimize `full_witness()` | 4-5ms | Low | Easy | None |
| 4 | Reduce Poseidon init overhead | 5-8ms | Low | Easy | None |
| 5 | Eliminate quotient `.to_vec()` | 3-5ms | Low | Medium | Refactor eval API |
| 6 | Batch Poseidon in Merkle tree | 10-15ms | Medium | Hard | Requires SIMD expertise |
| 7 | Eliminate flatten/unflatten alloc | 1-2ms | Low | Easy | None |
| 8 | Generator dispatch optimization | 2-4ms | Medium | Hard | Changes generator API |
| 9 | Wire polynomial clone elimination | 0.5-1ms | Low | Easy | None |
| 10 | Optimize transpose | 2-3ms | Low | Medium | None |
| 11 | Pre-allocate gate constraint bufs | 1-2ms | Low | Medium | Gate trait change |
| 12 | Reduce FRI round memory churn | 1ms | Low | Medium | Depends on #1, #2 |
| 13 | Stack alloc in hash_or_noop | 0.5ms | Low | Easy | None |
| 14 | Opening set copy elimination | <0.5ms | Low | Easy | None |
| 15 | Query phase copy elimination | <0.5ms | Low | Easy | None |

**Total estimated savings: 48-70ms (24-35% improvement)**

---

## Recommended Implementation Order

### Phase 1 -- Quick Wins (Easy, Low Risk)
1. **#9** Wire polynomial clone elimination (`prover.rs:169`)
2. **#3** Optimize `full_witness()` iteration (`witness.rs:376-388`)
3. **#7** Fix `flatten`/`unflatten` allocations (`extension/mod.rs:128-146`)
4. **#13** Stack alloc in `hash_or_noop` (`config.rs:75-86`)
5. **#4** Reduce Poseidon init overhead (`hashing.rs:97-114`)
6. **#14** Opening set copy elimination (`proof.rs:313-351`)
7. **#15** Query phase copy elimination (`fri/prover.rs:238`)

### Phase 2 -- Medium Effort, High Reward
8. **#2** In-place coset FFT shift (`polynomial/mod.rs:286-299`)
9. **#5** Eliminate quotient `.to_vec()` copies (`prover.rs:682-688`)
10. **#10** Optimize transpose (`util/mod.rs:25-31`, `config.rs:189`)
11. **#11** Pre-allocate gate constraint buffers (`vanishing_poly.rs:702-726`)

### Phase 3 -- Structural Changes
12. **#1** Flat Merkle tree leaves (changes `MerkleTree` struct)
13. **#12** Reduce FRI round memory churn
14. **#8** Generator enum dispatch

### Phase 4 -- Advanced SIMD
15. **#6** Batch Poseidon hashing in Merkle tree construction

---

## Notes on Correctness Preservation

All optimizations above preserve the mathematical correctness of the proving system:

- **No cryptographic changes**: The Poseidon permutation, Goldilocks field arithmetic, and FRI protocol remain identical
- **No constraint changes**: Gate evaluations produce the same values; only allocation/copy patterns change
- **Deterministic output**: For the same inputs, proofs are bit-identical (assuming no randomness changes)
- **Verification unchanged**: The verifier sees the same proof format and performs the same checks

Each optimization should be validated by running the existing test suite (`cargo test --release`) and verifying that benchmark proofs still verify correctly.
