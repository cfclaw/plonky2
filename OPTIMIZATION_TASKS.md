# Plonky2 Optimization Implementation Tasks

Concrete, step-by-step tasks for each optimization. Excludes batch Poseidon SIMD (#6 from original report). Ordered by recommended implementation sequence.

---

## Phase 1: Quick Wins

---

### Task 1: Wire Polynomial Clone Elimination

**Est. savings: 0.5-1ms | Difficulty: Easy | Risk: Low**

**Goal:** Eliminate 135 unnecessary `Vec<F>` clones when converting `MatrixWitness` to polynomial values.

**Steps:**

1. Open `plonky2/src/plonk/prover.rs:163-171`. Change:
   ```rust
   witness
       .wire_values
       .par_iter()
       .map(|column| PolynomialValues::new(column.clone()))
       .collect()
   ```
   to:
   ```rust
   witness
       .wire_values
       .into_par_iter()
       .map(PolynomialValues::new)
       .collect()
   ```

2. This requires `witness` (a `MatrixWitness`) to be consumed. Check that `witness` is not used after this point in `prove()`. It is not -- it's a local binding from `partition_witness.full_witness()` at line 155, only used here.

3. If `into_par_iter()` is not available for `Vec<Vec<F>>`, use `witness.wire_values.into_iter().map(...).collect()` -- the iteration itself isn't the bottleneck, the clone is.

**Files to change:**
- `plonky2/src/plonk/prover.rs` (line 169)

**Validation:** `cargo test --release -p plonky2`

---

### Task 2: Optimize `full_witness()` Iteration

**Est. savings: 4-5ms | Difficulty: Easy | Risk: Low**

**Goal:** Replace the O(degree x num_wires) brute-force scan with a direct iteration over populated values.

**Steps:**

1. Open `plonky2/src/iop/witness.rs:376-388`. The current code iterates all `(row, wire)` pairs even though most values are `None`. Replace with:

   ```rust
   pub fn full_witness(self) -> MatrixWitness<F> {
       let mut wire_values = vec![vec![F::ZERO; self.degree]; self.num_wires];
       let wire_range = self.degree * self.num_wires;
       for (rep_idx, value) in self.values.iter().enumerate() {
           if let Some(&v) = value.as_ref() {
               // Find all targets that map to this representative
               // and are wire targets (index < wire_range)
           }
       }
       MatrixWitness { wire_values }
   }
   ```

   However, the `representative_map` is many-to-one (multiple targets -> one representative). The simpler fix is to iterate the representative_map directly for wire targets only:

   ```rust
   pub fn full_witness(self) -> MatrixWitness<F> {
       let mut wire_values = vec![vec![F::ZERO; self.degree]; self.num_wires];
       for i in 0..self.degree {
           for j in 0..self.num_wires {
               let idx = i * self.num_wires + j;
               let rep = self.representative_map[idx];
               if let Some(v) = self.values[rep] {
                   wire_values[j][i] = v;
               }
           }
       }
       MatrixWitness { wire_values }
   }
   ```

   This eliminates the `Target::Wire` construction, `target.index()` multiplication/branch, and `try_get_target` method call overhead per iteration. The index calculation `i * self.num_wires + j` is the same computation that `target.index()` does internally (see `target.rs`) but without the `match` and without going through a method.

2. For further improvement, change the loop order to iterate in memory order of `representative_map` (row-major: row then wire), which it already is.

**Files to change:**
- `plonky2/src/iop/witness.rs` (lines 376-388)

**Validation:** `cargo test --release -p plonky2`

---

### Task 3: Fix `flatten`/`unflatten` Per-Element Allocations

**Est. savings: 1-2ms | Difficulty: Easy | Risk: Low**

**Goal:** Remove per-element `.to_vec()` allocations in extension field flatten/unflatten.

**Steps:**

1. Open `field/src/extension/mod.rs:128-135`. Change `flatten`:
   ```rust
   pub fn flatten<F, const D: usize>(l: &[F::Extension]) -> Vec<F>
   where
       F: Field + Extendable<D>,
   {
       let mut result = Vec::with_capacity(l.len() * D);
       for x in l {
           result.extend_from_slice(&x.to_basefield_array());
       }
       result
   }
   ```
   The key change: `extend_from_slice` on a fixed-size array reference instead of `.flat_map(|x| x.to_basefield_array().to_vec())`.

2. Open `field/src/extension/mod.rs:138-146`. Change `unflatten`:
   ```rust
   pub fn unflatten<F, const D: usize>(l: &[F]) -> Vec<F::Extension>
   where
       F: Field + Extendable<D>,
   {
       debug_assert_eq!(l.len() % D, 0);
       l.chunks_exact(D)
           .map(|c| {
               let arr: [F; D] = c.try_into().unwrap();
               F::Extension::from_basefield_array(arr)
           })
           .collect()
   }
   ```
   The key change: `c.try_into().unwrap()` converts the slice directly to `[F; D]` without going through a `Vec`.

   **Note:** This requires that `from_basefield_array` accepts `[F; D]` not `Vec`. Check the signature -- it takes `core::array::IntoIter` or array. If it takes a `Vec`, add an array-based overload or adjust accordingly. Looking at `Extendable` trait, `from_basefield_array` typically takes `[F; D]` so this should work.

**Files to change:**
- `field/src/extension/mod.rs` (lines 128-146)

**Validation:** `cargo test --release -p plonky2_field && cargo test --release -p plonky2`

---

### Task 4: Stack Allocation in `hash_or_noop`

**Est. savings: ~0.5ms | Difficulty: Easy | Risk: Low**

**Goal:** Replace heap `Vec` with stack array for small-input hashing.

**Steps:**

1. Find all implementations of `hash_or_noop`. The trait is in `plonky2/src/plonk/config.rs`. Search for `fn hash_or_noop` across the codebase. The primary implementation is in the `Hasher` trait default or in concrete types like `PoseidonHash`.

2. In each implementation, change the small-input path from:
   ```rust
   let mut inputs_bytes = vec![0u8; Self::HASH_SIZE];
   ```
   to:
   ```rust
   let mut inputs_bytes = [0u8; 32]; // HASH_SIZE is 32 for Poseidon/Goldilocks
   ```
   If `HASH_SIZE` is not const-evaluable, use a fixed size that covers the max (32 bytes) or use a `MaybeUninit<[u8; 32]>` approach.

**Files to change:**
- Search for all `hash_or_noop` implementations (likely `plonky2/src/hash/poseidon.rs`, `plonky2/src/hash/keccak.rs`, and trait default in `plonky2/src/plonk/config.rs`)

**Validation:** `cargo test --release -p plonky2`

---

### Task 5: Reduce Poseidon Permutation Init in `compress`

**Est. savings: 5-8ms | Difficulty: Easy | Risk: Low**

**Goal:** Minimize work in the `compress` function that's called for every Merkle tree internal node.

**Steps:**

1. Open `plonky2/src/hash/hashing.rs:97-114`. The `compress` function currently does:
   ```rust
   let mut perm = P::new(core::iter::repeat(F::ZERO));  // zeros all 12 elements
   perm.set_from_slice(&x.elements, 0);                 // overwrites elements 0..4
   perm.set_from_slice(&y.elements, NUM_HASH_OUT_ELTS);  // overwrites elements 4..8
   ```
   Elements 8..12 (the capacity) are zero. Elements 0..8 are overwritten. So the `P::new(zeros)` only needs to zero elements 8..12.

2. Add a `new_zeroed_capacity` or similar method to `PlonkyPermutation` that only zeros the capacity portion, or just set elements 8..12 explicitly after construction:
   ```rust
   pub fn compress<F: Field, P: PlonkyPermutation<F>>(x: HashOut<F>, y: HashOut<F>) -> HashOut<F> {
       let mut perm = P::new(core::iter::repeat(F::ZERO));
       // The above zeros all elements. We could optimize by only zeroing capacity.
       // For now, the main win is ensuring hash_n_to_m_no_pad uses hash_n_to_hash_no_pad.
       perm.set_from_slice(&x.elements, 0);
       perm.set_from_slice(&y.elements, NUM_HASH_OUT_ELTS);
       perm.permute();
       HashOut {
           elements: perm.squeeze()[..NUM_HASH_OUT_ELTS].try_into().unwrap(),
       }
   }
   ```

3. The bigger win is in `hash_n_to_m_no_pad` (line 118-141). It allocates `Vec::new()` for outputs. Ensure that all Merkle-tree leaf hashing paths call `hash_n_to_hash_no_pad` (line 143-145) which already exists and avoids the Vec:
   ```rust
   pub fn hash_n_to_hash_no_pad<F: RichField, P: PlonkyPermutation<F>>(inputs: &[F]) -> HashOut<F> {
       HashOut::from_vec(hash_n_to_m_no_pad::<F, P>(inputs, NUM_HASH_OUT_ELTS))
   }
   ```
   But this still calls `hash_n_to_m_no_pad` internally. Rewrite `hash_n_to_hash_no_pad` to avoid the Vec entirely:
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

4. Check that `PoseidonHash::hash_no_pad` (used by `hash_or_noop` on the large-input path) calls `hash_n_to_hash_no_pad`, not `hash_n_to_m_no_pad`. If it calls the Vec-returning version, update it.

**Files to change:**
- `plonky2/src/hash/hashing.rs` (lines 97-145)
- Possibly `plonky2/src/hash/poseidon.rs` (check `hash_no_pad` impl)

**Validation:** `cargo test --release -p plonky2`

---

### Task 6: Opening Set Copy Elimination

**Est. savings: <0.5ms | Difficulty: Easy | Risk: Low**

**Goal:** Avoid 8 `.to_vec()` calls in `OpeningSet::new()` by evaluating polynomials into targeted ranges directly.

**Steps:**

1. Open `plonky2/src/plonk/proof.rs:313-351`. Currently `eval_commitment` evaluates ALL polynomials in a batch and returns one big `Vec`, then we slice+clone subranges:
   ```rust
   let constants_sigmas_eval = eval_commitment(zeta, constants_sigmas_commitment);
   // ...
   constants: constants_sigmas_eval[common_data.constants_range()].to_vec(),
   plonk_sigmas: constants_sigmas_eval[common_data.sigmas_range()].to_vec(),
   ```

2. Replace with targeted evaluation. Instead of evaluating all polys then slicing, evaluate the ranges directly:
   ```rust
   let eval_range = |z: F::Extension, c: &PolynomialBatch<F, C, D>, range: Range<usize>| {
       c.polynomials[range]
           .par_iter()
           .map(|p| p.to_extension().eval(z))
           .collect::<Vec<_>>()
   };

   Self {
       constants: eval_range(zeta, constants_sigmas_commitment, common_data.constants_range()),
       plonk_sigmas: eval_range(zeta, constants_sigmas_commitment, common_data.sigmas_range()),
       wires: eval_commitment(zeta, wires_commitment),
       plonk_zs: eval_range(zeta, zs_partial_products_lookup_commitment, common_data.zs_range()),
       // etc.
   }
   ```

   **Alternative simpler approach:** Keep `eval_commitment` but destructure the Vec using `drain` or `split_off` instead of `.to_vec()` on slices. Since `constants_range` and `sigmas_range` partition the full `constants_sigmas_eval`, you can split the vec:
   ```rust
   let mut constants_sigmas_eval = eval_commitment(zeta, constants_sigmas_commitment);
   let plonk_sigmas = constants_sigmas_eval.split_off(common_data.constants_range().end);
   let constants = constants_sigmas_eval;
   // plonk_sigmas now starts at the right offset
   ```
   This is trickier to get right with overlapping ranges, so the `eval_range` approach is cleaner.

**Files to change:**
- `plonky2/src/plonk/proof.rs` (lines 313-351)

**Validation:** `cargo test --release -p plonky2`

---

### Task 7: FRI Query Phase `.to_vec()` Elimination

**Est. savings: <0.5ms | Difficulty: Easy | Risk: Low**

**Goal:** Avoid copying leaf data during FRI query round construction.

**Steps:**

1. Open `plonky2/src/fri/prover.rs:236-239`:
   ```rust
   let initial_proof = initial_merkle_trees
       .iter()
       .map(|t| (t.get(x_index).to_vec(), t.prove(x_index)))
       .collect::<Vec<_>>();
   ```

2. The `.to_vec()` is needed because the proof struct owns the data. Check `FriInitialTreeProof` to see if it stores `Vec<F>` (owned):
   - If yes, the `.to_vec()` is required for ownership transfer. In that case, skip this task or consider changing the proof type to hold `Cow<'a, [F]>` or `&'a [F]` -- but this adds lifetime complexity to the proof struct.
   - If the Merkle tree outlives the proof assembly, you could borrow. But since proofs are returned and trees are local, ownership is needed.

3. **Practical approach:** This copy is per query round (28 rounds) and per initial tree (~4 trees). Each copy is one leaf (~135 elements). Total: 28 x 4 x 135 x 8 bytes = ~120KB. This is genuinely small, so mark this as "nice to have" and skip if lifetime changes would be invasive.

**Files to change:**
- `plonky2/src/fri/prover.rs` (line 238) -- only if proof struct can hold borrows

**Validation:** `cargo test --release -p plonky2`

---

## Phase 2: Medium Effort, High Reward

---

### Task 8: In-Place Coset FFT Shift

**Est. savings: ~8ms | Difficulty: Medium | Risk: Low**

**Goal:** Eliminate the full polynomial copy in `coset_fft_with_options` by applying the shift in-place.

**Steps:**

1. Open `field/src/polynomial/mod.rs:286-299`. Add a new mutable method:
   ```rust
   pub fn coset_fft_with_options_mut(
       &mut self,
       shift: F,
       zero_factor: Option<usize>,
       root_table: Option<&FftRootTable<F>>,
   ) -> PolynomialValues<F> {
       // Apply shift in-place
       let mut s_pow = F::ONE;
       for c in self.coeffs.iter_mut() {
           *c *= s_pow;
           s_pow *= shift;
       }
       // Now do FFT (which is also in-place in fft_classic)
       self.fft_with_options(zero_factor, root_table)
   }
   ```

2. **Important:** `fft_with_options` takes `&self`, not `&mut self`. Look at its implementation -- it calls `fft_with_options` which creates a `PolynomialValues` from the coeffs. The FFT itself (`fft_dispatch` in `field/src/fft.rs`) operates on `&mut [F]`. So `fft_with_options` does:
   ```rust
   pub fn fft_with_options(...) -> PolynomialValues<F> {
       let mut values = self.coeffs.clone(); // Another clone!
       fft_dispatch(&mut values, ...);
       PolynomialValues::new(values)
   }
   ```
   We need a consuming variant that avoids this clone too:
   ```rust
   pub fn fft_with_options_consume(
       mut self,
       zero_factor: Option<usize>,
       root_table: Option<&FftRootTable<F>>,
   ) -> PolynomialValues<F> {
       fft_dispatch(&mut self.coeffs, zero_factor, root_table);
       PolynomialValues { values: self.coeffs }
   }
   ```

3. Combine into a consuming `coset_fft`:
   ```rust
   pub fn coset_fft_consume(
       mut self,
       shift: F,
       zero_factor: Option<usize>,
       root_table: Option<&FftRootTable<F>>,
   ) -> PolynomialValues<F> {
       let mut s_pow = F::ONE;
       for c in self.coeffs.iter_mut() {
           *c *= s_pow;
           s_pow *= shift;
       }
       fft_dispatch(&mut self.coeffs, zero_factor, root_table);
       PolynomialValues { values: self.coeffs }
   }
   ```

4. Update call sites to use the consuming variant where the polynomial is not reused afterward:
   - `plonky2/src/plonk/config.rs:177-179` -- In `compute_from_coeffs`, the polynomial `p` is from `polynomials.par_iter()`, so it's borrowed. You'd need `.clone()` here anyway... **unless** you change the pipeline to consume the polynomials. Check if `polynomials` is used after LDE generation. Looking at `compute_from_coeffs` (line 156-204), `polynomials` is stored in the returned `PolynomialBatch` at line 198. So we can't consume them.
   - **Better approach:** For `compute_from_coeffs`, the `lde()` call (line 177) already creates a new padded copy. The `coset_fft_with_options` then creates ANOTHER copy for shift. The fix is: `p.lde(rate_bits)` returns an owned `PolynomialCoeffs`. Call the consuming `coset_fft_consume` on that:
     ```rust
     .map(|p| {
         p.lde(rate_bits)
             .coset_fft_consume(F::coset_shift(), Some(rate_bits), fft_root_table)
             .values
     })
     ```
   - `plonky2/src/fri/prover.rs:119` -- `coeffs.coset_fft(shift.into())`. Here `coeffs` is reassigned on the next loop iteration, so consuming is fine. Use `coeffs.coset_fft_consume(shift.into(), None, None)`.
   - `plonky2/src/plonk/config.rs:247` -- `values.coset_ifft(F::coset_shift())`. Same pattern for IFFT -- add a consuming `coset_ifft_consume`.

5. Also add `fft_with_options` on `PolynomialValues` to verify the existing path for `fft_with_options` on `PolynomialCoeffs`. Check `field/src/polynomial/mod.rs` for the exact FFT entry point and ensure the consuming variant plugs in correctly.

**Files to change:**
- `field/src/polynomial/mod.rs` (add consuming methods, ~20 lines)
- `plonky2/src/plonk/config.rs` (lines 177-179, 247)
- `plonky2/src/fri/prover.rs` (line 119)

**Validation:** `cargo test --release -p plonky2_field && cargo test --release -p plonky2`

---

### Task 9: Eliminate Quotient Batch `.to_vec()` Copies

**Est. savings: 3-5ms | Difficulty: Medium | Risk: Low**

**Goal:** Replace owned `Vec` intermediaries with direct slice references into LDE Merkle tree leaves.

**Steps:**

1. Open `plonky2/src/plonk/prover.rs:662-704`. The pattern is:
   ```rust
   // For each point k in the batch:
   let local_zs_partial_and_lookup = commitment.get_lde_values(i, step); // returns &[F]
   local_zs_batch_vecs.push(local_zs_partial_and_lookup[range].to_vec());
   // Later:
   let local_zs_batch: Vec<&[F]> = local_zs_batch_vecs.iter().map(|v| v.as_slice()).collect();
   ```
   The `.to_vec()` + `as_slice()` round-trip is unnecessary. The original `get_lde_values` returns a `&[F]` that lives as long as the commitment, which outlives the batch.

2. Replace with direct sub-slice references. Instead of copying into `_vecs` then re-borrowing:
   ```rust
   // Before the loop, prepare containers for references:
   let mut s_sigmas_batch: Vec<&[F]> = Vec::with_capacity(n);
   let mut local_zs_batch: Vec<&[F]> = Vec::with_capacity(n);
   let mut next_zs_batch: Vec<&[F]> = Vec::with_capacity(n);
   let mut partial_products_batch: Vec<&[F]> = Vec::with_capacity(n);
   let mut local_lookup_batch: Vec<&[F]> = Vec::with_capacity(n);
   let mut next_lookup_batch: Vec<&[F]> = Vec::with_capacity(n);

   for (k, (&i, &x)) in indices_batch.iter().zip(xs_batch).enumerate() {
       // ... existing code for shifted_xs_batch, constants, wires ...

       let local_constants_sigmas = prover_data.constants_sigmas_commitment.get_lde_values(i, step);
       let local_zs_partial_and_lookup = zs_partial_products_and_lookup_commitment.get_lde_values(i, step);
       let next_zs_partial_and_lookup = zs_partial_products_and_lookup_commitment.get_lde_values(i_next, step);

       s_sigmas_batch.push(&local_constants_sigmas[common_data.sigmas_range()]);
       local_zs_batch.push(&local_zs_partial_and_lookup[common_data.zs_range()]);
       next_zs_batch.push(&next_zs_partial_and_lookup[common_data.zs_range()]);
       partial_products_batch.push(&local_zs_partial_and_lookup[common_data.partial_products_range()]);
       if has_lookup {
           local_lookup_batch.push(&local_zs_partial_and_lookup[common_data.lookup_range()]);
           next_lookup_batch.push(&next_zs_partial_and_lookup[common_data.lookup_range()]);
       }
   }
   ```

3. Remove the `_vecs` intermediaries (lines 663-668) and the conversion lines (699-704).

4. This compiles because `prover_data`, `wires_commitment`, and `zs_partial_products_and_lookup_commitment` all outlive the batch closure -- they're passed as references to `compute_quotient_polys_pre_transpose`.

**Files to change:**
- `plonky2/src/plonk/prover.rs` (lines 662-704)

**Validation:** `cargo test --release -p plonky2`

---

### Task 10: Tiled Transpose for LDE Values

**Est. savings: 2-3ms | Difficulty: Medium | Risk: Low**

**Goal:** Replace cache-unfriendly column-wise transpose with a tiled approach.

**Steps:**

1. Open `plonky2/src/util/mod.rs:25-31`. The current transpose:
   ```rust
   pub fn transpose<T: Send + Sync + Copy>(matrix: &[Vec<T>]) -> Vec<Vec<T>> {
       let len = matrix[0].len();
       (0..len)
           .into_par_iter()
           .map(|i| matrix.iter().map(|row| row[i]).collect())
           .collect()
   }
   ```
   For each output row `i`, it reads element `i` from every input row -- scattered across memory.

2. Add a tiled transpose that processes blocks:
   ```rust
   pub fn transpose_tiled<T: Send + Sync + Copy + Default>(matrix: &[Vec<T>]) -> Vec<Vec<T>> {
       let nrows = matrix.len();
       if nrows == 0 { return vec![]; }
       let ncols = matrix[0].len();

       // Allocate output
       let mut result: Vec<Vec<T>> = (0..ncols)
           .map(|_| vec![T::default(); nrows])
           .collect();

       const TILE: usize = 64;

       // Process in tiles for cache friendliness
       for row_start in (0..nrows).step_by(TILE) {
           for col_start in (0..ncols).step_by(TILE) {
               let row_end = (row_start + TILE).min(nrows);
               let col_end = (col_start + TILE).min(ncols);
               for r in row_start..row_end {
                   for c in col_start..col_end {
                       result[c][r] = matrix[r][c];
                   }
               }
           }
       }
       result
   }
   ```

3. If Merkle tree leaves are changed to flat storage (Task 13), write a `transpose_to_flat` that outputs a single `Vec<T>` with stride, avoiding the inner `Vec` allocations entirely:
   ```rust
   pub fn transpose_to_flat<T: Send + Sync + Copy + Default>(
       matrix: &[Vec<T>],
   ) -> (Vec<T>, usize) {
       let nrows = matrix.len();
       let ncols = matrix[0].len();
       let mut flat = vec![T::default(); nrows * ncols];
       // tiled write into flat[col * nrows + row]
       // ...
       (flat, nrows)  // returns (data, leaf_len)
   }
   ```

4. Replace calls to `transpose()` in:
   - `plonky2/src/plonk/config.rs:189`
   - `plonky2/src/plonk/config.rs:244`
   - `plonky2/src/plonk/prover.rs` (partial products transpose)

**Files to change:**
- `plonky2/src/util/mod.rs` (add tiled variant, modify or replace existing `transpose`)
- `plonky2/src/plonk/config.rs` (call sites)

**Validation:** `cargo test --release -p plonky2`

---

### Task 11: Pre-Allocate Gate Constraint Buffers

**Est. savings: 1-2ms | Difficulty: Medium | Risk: Low**

**Goal:** Eliminate per-gate-per-batch Vec allocation in constraint evaluation.

**Steps:**

1. Open `plonky2/src/gates/gate.rs:111-120`. Currently:
   ```rust
   fn eval_unfiltered_base_batch(&self, vars_base: EvaluationVarsBaseBatch<F>) -> Vec<F> {
       let mut res = vec![F::ZERO; vars_base.len() * self.num_constraints()];
       for (i, vars_base_one) in vars_base.iter().enumerate() {
           self.eval_unfiltered_base_one(vars_base_one, StridedConstraintConsumer::new(&mut res, vars_base.len(), i));
       }
       res
   }
   ```

2. Change the signature to write into a pre-allocated buffer:
   ```rust
   fn eval_unfiltered_base_batch_into(
       &self,
       vars_base: EvaluationVarsBaseBatch<F>,
       output: &mut [F],  // pre-allocated, caller provides
   ) {
       // Zero the relevant portion
       output[..vars_base.len() * self.num_constraints()].fill(F::ZERO);
       for (i, vars_base_one) in vars_base.iter().enumerate() {
           self.eval_unfiltered_base_one(
               vars_base_one,
               StridedConstraintConsumer::new(output, vars_base.len(), i),
           );
       }
   }
   ```

3. Update `evaluate_gate_constraints_base_batch` in `vanishing_poly.rs:702-728`:
   ```rust
   pub fn evaluate_gate_constraints_base_batch<F: RichField + Extendable<D>, const D: usize>(
       common_data: &CommonCircuitData<F, D>,
       vars_batch: EvaluationVarsBaseBatch<F>,
   ) -> Vec<F> {
       let mut constraints_batch = vec![F::ZERO; common_data.num_gate_constraints * vars_batch.len()];
       let mut gate_buf = vec![F::ZERO; max_gate_constraints * vars_batch.len()];

       for (i, gate) in common_data.gates.iter().enumerate() {
           gate.0.eval_unfiltered_base_batch_into(vars_batch, &mut gate_buf);
           // Apply filter and accumulate into constraints_batch
           // ... (existing filter + batch_add_inplace logic)
       }
       constraints_batch
   }
   ```
   Pre-compute `max_gate_constraints` as the max of `gate.num_constraints()` across all gates. Allocate `gate_buf` once outside the gate loop.

4. Update `eval_filtered_base_batch` in `gate.rs:159-179` to work with the buffer approach. The filter multiplication and accumulation can happen on the buffer without allocating.

**Files to change:**
- `plonky2/src/gates/gate.rs` (add `eval_unfiltered_base_batch_into` method)
- `plonky2/src/plonk/vanishing_poly.rs` (lines 702-728)

**Validation:** `cargo test --release -p plonky2`

---

## Phase 3: Structural Changes

---

### Task 12: Flat Merkle Tree Leaf Storage

**Est. savings: 10-12ms | Difficulty: Medium-Hard | Risk: Medium (public API change)**

**Goal:** Replace `Vec<Vec<F>>` with contiguous `Vec<F>` for Merkle tree leaves.

**Steps:**

1. Open `plonky2/src/hash/merkle_tree.rs:51-67`. Change:
   ```rust
   pub struct MerkleTree<F: RichField, H: Hasher<F>> {
       pub leaves: Vec<F>,        // flat contiguous storage
       pub leaf_len: usize,       // elements per leaf
       pub digests: Vec<H::Hash>,
       pub cap: MerkleCap<F, H>,
   }
   ```

2. Update `MerkleTree::new` (line 198) to accept `(Vec<F>, usize)` or a new type:
   ```rust
   pub fn new(leaves: Vec<F>, leaf_len: usize, cap_height: usize) -> Self {
       let num_leaves = leaves.len() / leaf_len;
       // ...
   }
   ```

3. Update `get` (line 231):
   ```rust
   pub fn get(&self, i: usize) -> &[F] {
       &self.leaves[i * self.leaf_len..(i + 1) * self.leaf_len]
   }
   ```

4. Update `fill_subtree` (line 91-118) to work with flat storage. Change the `leaves` parameter from `&[Vec<F>]` to `(&[F], usize)` (flat slice + leaf_len):
   ```rust
   pub(crate) fn fill_subtree<F: RichField, H: Hasher<F>>(
       digests_buf: &mut [MaybeUninit<H::Hash>],
       leaves: &[F],
       leaf_len: usize,
   ) -> H::Hash {
       let num_leaves = leaves.len() / leaf_len;
       assert_eq!(num_leaves, digests_buf.len() / 2 + 1);
       if digests_buf.is_empty() {
           H::hash_or_noop(&leaves[..leaf_len])
       } else {
           // split leaves at midpoint
           let mid = (num_leaves / 2) * leaf_len;
           let (left_leaves, right_leaves) = leaves.split_at(mid);
           // ... rest same as before but with (left_leaves, leaf_len) etc.
       }
   }
   ```

5. Update `fill_digests_buf` (line 120-154) similarly.

6. Update `prove` (line 236-242) -- no change needed, it calls `self.leaves.len()` which becomes the flat length. Fix: use `self.leaves.len() / self.leaf_len` for num_leaves.

7. Update all callers that construct `MerkleTree::new(leaves, cap_height)`:
   - `plonky2/src/plonk/config.rs:195` -- `build_merkle_tree(timing, leaves, cap_height)`
   - `plonky2/src/fri/prover.rs:104` -- `MerkleTree::new(chunked_values, ...)`
   - `plonky2/src/batch_fri/oracle.rs` -- if used
   - Search for all `MerkleTree::new(` calls

8. Update the `transpose` call sites to produce flat output instead of `Vec<Vec<F>>` (see Task 10's `transpose_to_flat`).

9. Update `get_lde_values` in `fri/oracle.rs:110-115`:
   ```rust
   pub fn get_lde_values(&self, index: usize, step: usize) -> &[F] {
       let index = index * step;
       let index = reverse_bits(index, self.degree_log + self.rate_bits);
       self.merkle_tree.get(index)  // now returns &[F] from flat storage
       // trim salt as before
   }
   ```

10. Update `batch_merkle_tree.rs` if it exists and uses `Vec<Vec<F>>` leaves.

**Files to change:**
- `plonky2/src/hash/merkle_tree.rs` (struct + all methods)
- `plonky2/src/plonk/config.rs` (build_merkle_tree, transpose)
- `plonky2/src/fri/prover.rs` (tree construction)
- `plonky2/src/fri/oracle.rs` (get_lde_values)
- `plonky2/src/util/mod.rs` (transpose_to_flat)
- All other `MerkleTree::new` call sites
- Serialization code if MerkleTree is serialized

**Validation:** `cargo test --release -p plonky2` and run the benchmark to verify speedup.

---

### Task 13: Reduce FRI Round Memory Churn

**Est. savings: ~1ms | Difficulty: Medium | Risk: Low**

**Goal:** Minimize allocations and copies in the FRI committed tree loop.

**Steps:**

1. Open `plonky2/src/fri/prover.rs:84-120`. Three sub-optimizations:

   **a) Use consuming coset FFT (depends on Task 8):**
   Line 119: `values = coeffs.coset_fft(shift.into())` -- replace with `coeffs.coset_fft_consume(...)` to avoid the shift copy.

   **b) Use improved `flatten` (depends on Task 3):**
   Line 102: `.map(|chunk| flatten(chunk))` -- already fixed by Task 3 to avoid per-element `.to_vec()`.

   **c) Use flat Merkle tree (depends on Task 12):**
   Line 104: `MerkleTree::new(chunked_values, ...)` -- if leaves are flat, the chunked_values can be produced as a single flat Vec instead of Vec<Vec<F>>.

2. If Tasks 8 and 12 are done, the FRI loop naturally benefits. The main additional work is:
   - Producing `chunked_values` as a flat buffer: the `par_chunks(arity).map(flatten).collect()` can become a single flat allocation where each chunk's flattened output goes to the right offset.

**Files to change:**
- `plonky2/src/fri/prover.rs` (lines 95-120)

**Validation:** `cargo test --release -p plonky2`

---

### Task 14: Generator Dependency Pre-Resolution

**Est. savings: 2-4ms | Difficulty: Hard | Risk: Medium**

**Goal:** Reduce generator dispatch overhead by building a dependency DAG and eliminating repeated failed invocations.

**Steps:**

1. Open `plonky2/src/iop/generator.rs:40-97`. The current loop:
   ```
   while pending_generators is not empty:
       for each pending generator:
           try to run it
           if dependencies not met: re-queue
           if ran: drain buffer, wake dependents
   ```
   Generators that can't run (missing deps) are re-queued and retried next round. With ~8K generators and ~10-50 rounds, many generators are attempted multiple times before their inputs arrive.

2. **Build a dependency DAG at circuit build time:**
   - For each generator, record its `watch_list` (the targets it depends on)
   - Build a map: `target -> Vec<generator_index>` (already exists as `generator_indices_by_watches`)
   - Instead of retrying all pending generators each round, only attempt generators whose watch targets have been newly satisfied

3. The existing code already uses `generator_indices_by_watches` for wakeup. The optimization is to **not re-queue generators that haven't been woken** -- i.e., only iterate newly-woken generators each round rather than all pending ones.

4. Replace the outer `pending_generator_indices` loop with a queue-based approach:
   ```rust
   let mut ready_queue: VecDeque<usize> = VecDeque::new();
   // Initially, add all generators with no watches or whose watches are already set
   for (idx, gen) in generators.iter().enumerate() {
       if gen.watch_list().is_empty() || witness.contains_all(&gen.watch_list()) {
           ready_queue.push_back(idx);
       }
   }

   while let Some(gen_idx) = ready_queue.pop_front() {
       let finished = generators[gen_idx].0.run(&witness, &mut buffer);
       // drain buffer, update witness
       // for each newly-set target, wake dependent generators
       for woken_gen in newly_woken {
           ready_queue.push_back(woken_gen);
       }
   }
   ```

5. **Sort generators by type** for better branch prediction. After building the dependency DAG, within each "ready" batch, sort by discriminant/type so the CPU branch predictor can learn the vtable pattern.

**Files to change:**
- `plonky2/src/iop/generator.rs` (lines 40-97)

**Validation:** `cargo test --release -p plonky2`

---

## Validation Checklist

After implementing any task:

1. `cargo test --release -p plonky2_field` -- field arithmetic correctness
2. `cargo test --release -p plonky2` -- full proving system correctness
3. `cargo run --release --example psy_bench_recursion --package plonky2 -- -vv` -- benchmark to measure improvement
4. Verify proofs still verify: the benchmark does this automatically ("Verified 100 times successfully")
5. Check for regressions: compare timing output line by line against the baseline trace
