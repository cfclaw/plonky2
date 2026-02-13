use crate::context::WebGpuContext;
use crate::utils;
use anyhow::Result;
use plonky2::field::extension::quadratic::QuadraticExtension;
use plonky2::field::fft::FftRootTable;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use plonky2::field::types::{Field, Sample};
use plonky2::fri::oracle::PolynomialBatch;
use plonky2::hash::hash_types::HashOut;
use plonky2::hash::merkle_tree::{MerkleCap, MerkleTree};
use plonky2::hash::poseidon::PoseidonHash;
use plonky2::plonk::config::{CpuProverCompute, GenericConfig, ProverCompute};
use plonky2::timed;
use plonky2::util::timing::TimingTree;
use plonky2::util::log2_strict;
use plonky2_maybe_rayon::*;
use serde::Serialize;

#[derive(Debug, Copy, Clone, Default, Eq, PartialEq, Serialize)]
pub struct PoseidonGoldilocksWebGpuConfig;

impl GenericConfig<2> for PoseidonGoldilocksWebGpuConfig {
    type F = GoldilocksField;
    type FE = QuadraticExtension<Self::F>;
    type Hasher = PoseidonHash;
    type InnerHasher = PoseidonHash;
    type Compute = WebGpuProverCompute;
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct WebGpuProverCompute;

mod fft {
    use super::*;

    /// Params struct matching the WGSL FftParams layout (8 x u32 = 32 bytes)
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct FftParams {
        n: u32,
        log_n: u32,
        num_coeffs: u32,
        layer: u32,
        stride: u32,
        n_inv_lo: u32,
        n_inv_hi: u32,
        twiddle_offset: u32,
    }

    /// Params for the Montgomery conversion shader
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    pub(super) struct MontConvertParams {
        n: u32,
        num_polys: u32,
        direction: u32, // 0 = to_mont, 1 = from_mont
        _pad: u32,
    }

    /// Params for the transpose shader
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    pub(super) struct TransposeParams {
        lde_size: u32,
        num_polys: u32,
        num_cols: u32,
        log_lde_size: u32,
    }

    /// Encode a Montgomery conversion pass into the command encoder.
    /// Converts the buffer in-place: direction 0 = to_mont, 1 = from_mont.
    pub(super) fn encode_mont_convert(
        ctx: &WebGpuContext,
        encoder: &mut wgpu::CommandEncoder,
        buffer: &wgpu::Buffer,
        n: usize,
        num_polys: usize,
        direction: u32,
    ) {
        let pipeline = ctx.get_pipeline("mont_convert_batched").unwrap();

        let params = MontConvertParams {
            n: n as u32,
            num_polys: num_polys as u32,
            direction,
            _pad: 0,
        };
        let params_buffer = super::create_uniform_buffer(&ctx.device, &params);

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mont_convert_bg"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("mont_convert_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        let wg_x = utils::div_ceil(n as u32, 256);
        pass.dispatch_workgroups(wg_x, num_polys as u32, 1);
        drop(pass);
    }

    /// Encode a transpose + bit_reverse pass into the command encoder.
    /// Transposes from polynomial-major (src) to evaluation-major with bit-reversed rows (dst).
    pub(super) fn encode_transpose_and_bit_reverse(
        ctx: &WebGpuContext,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::Buffer,
        dst: &wgpu::Buffer,
        lde_size: usize,
        num_polys: usize,
        num_cols: usize,
        log_lde_size: usize,
    ) {
        let pipeline = ctx.get_pipeline("transpose_and_bit_reverse").unwrap();

        let params = TransposeParams {
            lde_size: lde_size as u32,
            num_polys: num_polys as u32,
            num_cols: num_cols as u32,
            log_lde_size: log_lde_size as u32,
        };
        let params_buffer = super::create_uniform_buffer(&ctx.device, &params);

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("transpose_bg"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: src.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dst.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("transpose_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        let wg_x = utils::div_ceil(lde_size as u32, 256);
        pass.dispatch_workgroups(wg_x, num_polys as u32, 1);
        drop(pass);
    }

    /// Upload raw GoldilocksField data from multiple slices into a contiguous GPU buffer.
    /// Returns the buffer with elements as raw u64 bytes (standard form, not Montgomery).
    fn upload_raw_field_slices(
        ctx: &WebGpuContext,
        slices: &[&[GoldilocksField]],
        elements_per_slice: usize,
    ) -> wgpu::Buffer {
        let num_slices = slices.len();
        let total_bytes = (num_slices * elements_per_slice * 8) as u64;

        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("field_upload"),
            size: total_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let mut mapped = buffer.slice(..).get_mapped_range_mut();
            for (i, slice) in slices.iter().enumerate() {
                let offset = i * elements_per_slice * 8;
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        slice.as_ptr() as *const u8,
                        elements_per_slice * 8,
                    )
                };
                mapped[offset..offset + elements_per_slice * 8].copy_from_slice(bytes);
            }
        }
        buffer.unmap();
        buffer
    }

    /// Encode batched FFT DIT passes into the command encoder.
    fn encode_fft_passes_batched(
        ctx: &WebGpuContext,
        encoder: &mut wgpu::CommandEncoder,
        buffer: &wgpu::Buffer,
        log_n: usize,
        stride: usize,
        num_polys: usize,
        twiddles: &wgpu::Buffer,
    ) {
        let pipeline = ctx.get_pipeline("fft_dit_pass_batched").unwrap();
        let n = 1usize << log_n;
        let mut twiddle_elem_offset: u32 = 0;

        for layer in 0..log_n {
            let params = FftParams {
                n: n as u32,
                log_n: log_n as u32,
                num_coeffs: 0,
                layer: layer as u32,
                stride: stride as u32,
                n_inv_lo: 0,
                n_inv_hi: 0,
                twiddle_offset: twiddle_elem_offset,
            };

            let params_buffer = super::create_uniform_buffer(&ctx.device, &params);

            let bind_group_layout = pipeline.get_bind_group_layout(0);
            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fft_dit_bg"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: twiddles.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fft_dit_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);

            let wg_x = utils::div_ceil((n / 2) as u32, 256);
            pass.dispatch_workgroups(wg_x, num_polys as u32, 1);
            drop(pass);

            twiddle_elem_offset += 1u32 << layer;
        }
    }

    /// Core coset FFT encoding. Uploads coefficients, encodes to_mont + coset_scale + FFT passes.
    /// Returns (src_buffer, dst_buffer, shifts_buffer). After submission, dst_buffer contains
    /// FFT results still in Montgomery form. All three buffers must be explicitly destroyed
    /// by the caller after queue.submit() to free GPU memory.
    pub(super) fn encode_coset_fft_batched(
        ctx: &WebGpuContext,
        encoder: &mut wgpu::CommandEncoder,
        all_coeffs: &[&[GoldilocksField]],
        log_lde_size: usize,
        shift: GoldilocksField,
    ) -> Result<(wgpu::Buffer, wgpu::Buffer, wgpu::Buffer)> {
        let num_polys = all_coeffs.len();
        let num_coeffs = all_coeffs[0].len();
        let lde_size = 1 << log_lde_size;

        // Upload raw coefficients (no CPU-side Montgomery conversion)
        let src_buffer = upload_raw_field_slices(ctx, all_coeffs, num_coeffs);

        // GPU: convert to Montgomery form
        encode_mont_convert(ctx, encoder, &src_buffer, num_coeffs, num_polys, 0);

        // Upload shift powers (small, keep computing on CPU since it's just num_coeffs elements)
        let powers_of_shift: Vec<u64> = shift
            .powers()
            .take(num_coeffs)
            .map(|f| utils::to_mont(f.0))
            .collect();
        let shifts_buffer = utils::upload_u64_data(&ctx.device, &powers_of_shift, "fft_shifts");

        let dst_size = (num_polys * lde_size * 8) as u64;
        let dst_buffer = utils::create_storage_buffer(&ctx.device, dst_size, "fft_dst");

        // Coset scale + bit reverse params
        let params = FftParams {
            n: lde_size as u32,
            log_n: log_lde_size as u32,
            num_coeffs: num_coeffs as u32,
            layer: 0,
            stride: lde_size as u32,
            n_inv_lo: 0,
            n_inv_hi: 0,
            twiddle_offset: 0,
        };
        let params_buffer = super::create_uniform_buffer(&ctx.device, &params);

        // Encode coset scale + bit reverse
        {
            let pipeline = ctx
                .get_pipeline("coset_scale_and_bit_reverse_batched")
                .unwrap();
            let bind_group_layout = pipeline.get_bind_group_layout(0);
            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("coset_scale_bg"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: dst_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: src_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: shifts_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("coset_scale_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            let wg_x = utils::div_ceil(lde_size as u32, 256);
            pass.dispatch_workgroups(wg_x, num_polys as u32, 1);
            drop(pass);
        }

        // Encode FFT passes
        let twiddles = ctx
            .twiddle_buffers
            .get(&log_lde_size)
            .ok_or_else(|| anyhow::anyhow!("No twiddle buffer for log_n={}", log_lde_size))?;
        encode_fft_passes_batched(ctx, encoder, &dst_buffer, log_lde_size, lde_size, num_polys, twiddles);

        Ok((src_buffer, dst_buffer, shifts_buffer))
    }

    macro_rules! impl_fft_batch_functions {
        ($($async_kw:tt)*) => {
            /// Batch coset FFT: process all polynomials in a single GPU dispatch sequence.
            /// Uploads raw field elements and performs Montgomery conversion on GPU.
            pub $($async_kw)* fn run_gpu_coset_fft_batch(
                ctx: &WebGpuContext,
                all_coeffs: &[&[GoldilocksField]],
                log_lde_size: usize,
                shift: GoldilocksField,
            ) -> Result<Vec<Vec<GoldilocksField>>> {
                let num_polys = all_coeffs.len();
                if num_polys == 0 {
                    return Ok(vec![]);
                }

                let lde_size = 1 << log_lde_size;

                let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("fft_coset_encoder"),
                });

                let (src_buffer, dst_buffer, shifts_buffer) =
                    encode_coset_fft_batched(ctx, &mut encoder, all_coeffs, log_lde_size, shift)?;

                // GPU: convert from Montgomery form (in-place on dst_buffer)
                encode_mont_convert(ctx, &mut encoder, &dst_buffer, lde_size, num_polys, 1);

                ctx.queue.submit(Some(encoder.finish()));

                // Free GPU buffers that are no longer referenced by any pending commands.
                // The submitted commands will keep their internal references until complete.
                // This reduces peak memory during the download below.
                src_buffer.destroy();
                shifts_buffer.destroy();

                // Download results (already in standard form - no CPU from_mont needed)
                let all_data = plonky2::maybe_await!(
                    utils::download_field_data(ctx, &dst_buffer, num_polys * lde_size)
                );

                dst_buffer.destroy();

                // Split into per-polynomial vectors
                let results: Vec<Vec<GoldilocksField>> = (0..num_polys)
                    .into_par_iter()
                    .map(|i| {
                        let offset = i * lde_size;
                        all_data[offset..offset + lde_size].to_vec()
                    })
                    .collect();

                Ok(results)
            }

            /// Batch IFFT: process all polynomials in a single GPU dispatch sequence.
            /// Uploads raw field elements and performs Montgomery conversion on GPU.
            pub $($async_kw)* fn run_gpu_ifft_batch(
                ctx: &WebGpuContext,
                all_values: &[&[GoldilocksField]],
            ) -> Result<Vec<PolynomialCoeffs<GoldilocksField>>> {
                let num_polys = all_values.len();
                if num_polys == 0 {
                    return Ok(vec![]);
                }

                let n = all_values[0].len();
                let log_n = log2_strict(n);

                // Upload raw field elements (no CPU-side Montgomery conversion)
                let src_buffer = upload_raw_field_slices(ctx, all_values, n);

                let dst_size = (num_polys * n * 8) as u64;
                let dst_buffer = utils::create_storage_buffer(&ctx.device, dst_size, "ifft_dst");

                let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("ifft_encoder"),
                });

                // GPU: convert source to Montgomery form
                encode_mont_convert(ctx, &mut encoder, &src_buffer, n, num_polys, 0);

                // Step 1: Bit-reverse copy
                {
                    let params = FftParams {
                        n: n as u32,
                        log_n: log_n as u32,
                        num_coeffs: 0,
                        layer: 0,
                        stride: n as u32,
                        n_inv_lo: 0,
                        n_inv_hi: 0,
                        twiddle_offset: 0,
                    };
                    let params_buffer = super::create_uniform_buffer(&ctx.device, &params);

                    let pipeline = ctx.get_pipeline("bit_reverse_copy_batched").unwrap();
                    let bind_group_layout = pipeline.get_bind_group_layout(0);
                    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("br_bg"),
                        layout: &bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: dst_buffer.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 1, resource: src_buffer.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 2, resource: params_buffer.as_entire_binding() },
                        ],
                    });

                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("bit_reverse_pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, Some(&bind_group), &[]);
                    let wg_x = utils::div_ceil(n as u32, 256);
                    pass.dispatch_workgroups(wg_x, num_polys as u32, 1);
                    drop(pass);
                }

                // Step 2: FFT passes (forward twiddles)
                let twiddles = ctx.twiddle_buffers.get(&log_n)
                    .ok_or_else(|| anyhow::anyhow!("No twiddle buffer for log_n={}", log_n))?;
                encode_fft_passes_batched(ctx, &mut encoder, &dst_buffer, log_n, n, num_polys, twiddles);

                // Step 3: Reorder and scale
                {
                    let n_inv = GoldilocksField::inverse_2exp(log_n);
                    let n_inv_mont = utils::to_mont(n_inv.0);

                    let params = FftParams {
                        n: n as u32,
                        log_n: log_n as u32,
                        num_coeffs: 0,
                        layer: 0,
                        stride: n as u32,
                        n_inv_lo: n_inv_mont as u32,
                        n_inv_hi: (n_inv_mont >> 32) as u32,
                        twiddle_offset: 0,
                    };
                    let params_buffer = super::create_uniform_buffer(&ctx.device, &params);

                    let pipeline = ctx.get_pipeline("ifft_reorder_and_scale_batched").unwrap();
                    let bind_group_layout = pipeline.get_bind_group_layout(0);
                    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("ifft_reorder_bg"),
                        layout: &bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: dst_buffer.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 1, resource: params_buffer.as_entire_binding() },
                        ],
                    });

                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("ifft_reorder_pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, Some(&bind_group), &[]);
                    let wg_x = utils::div_ceil((n / 2) as u32, 256);
                    pass.dispatch_workgroups(wg_x, num_polys as u32, 1);
                    drop(pass);
                }

                // GPU: convert from Montgomery form (in-place on dst_buffer)
                encode_mont_convert(ctx, &mut encoder, &dst_buffer, n, num_polys, 1);

                ctx.queue.submit(Some(encoder.finish()));

                // Free src_buffer before download — no longer referenced by download commands.
                src_buffer.destroy();

                // Download results (already in standard form)
                let all_data = plonky2::maybe_await!(
                    utils::download_field_data(ctx, &dst_buffer, num_polys * n)
                );

                dst_buffer.destroy();

                // Split into PolynomialCoeffs (no from_mont needed)
                let results: Vec<PolynomialCoeffs<GoldilocksField>> = (0..num_polys)
                    .into_par_iter()
                    .map(|i| {
                        let offset = i * n;
                        PolynomialCoeffs::new(all_data[offset..offset + n].to_vec())
                    })
                    .collect();

                Ok(results)
            }
        };
    }

    #[cfg(not(feature = "async_prover"))]
    impl_fft_batch_functions!();

    #[cfg(feature = "async_prover")]
    impl_fft_batch_functions!(async);
}

/// Params struct matching the WGSL MerkleParams layout (8 x u32 = 32 bytes)
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MerkleParams {
    num_leaves: u32,
    leaf_size: u32,
    layer_idx: u32,
    pairs_per_subtree: u32,
    subtree_digest_len: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

fn create_uniform_buffer<T: bytemuck::Pod>(device: &wgpu::Device, data: &T) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("uniform_params"),
        size: std::mem::size_of::<T>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });
    buffer
        .slice(..)
        .get_mapped_range_mut()
        .copy_from_slice(bytemuck::bytes_of(data));
    buffer.unmap();
    buffer
}

type F = GoldilocksField;
type C = PoseidonGoldilocksWebGpuConfig;
const D: usize = 2;

macro_rules! impl_webgpu_prover_compute {
    ($($async_kw:tt)*) => {
        impl ProverCompute<F, C, D> for WebGpuProverCompute {
            $($async_kw)* fn build_merkle_tree(
                timing: &mut TimingTree,
                leaves: Vec<Vec<F>>,
                cap_height: usize,
            ) -> anyhow::Result<MerkleTree<F, PoseidonHash>> {
                let num_leaves = leaves.len();
                if num_leaves == 0 {
                    return plonky2::maybe_await!(
                        <CpuProverCompute as ProverCompute<F, C, D>>::build_merkle_tree(
                            timing, leaves, cap_height,
                        )
                    );
                }

                let log2_leaves = log2_strict(num_leaves);
                let leaf_size = leaves[0].len();

                // Fall back to CPU for very small trees
                if log2_leaves < 4 || leaf_size == 0 {
                    return plonky2::maybe_await!(
                        <CpuProverCompute as ProverCompute<F, C, D>>::build_merkle_tree(
                            timing, leaves, cap_height,
                        )
                    );
                }

                timed!(timing, "build Merkle tree (WebGPU)", {
                    let ctx_guard = crate::context::acquire_context()?;
                    let ctx = &*ctx_guard;

                    const HASH_SIZE: usize = 4; // NUM_HASH_OUT_ELTS

                    // Upload flattened leaf data directly from slices (no intermediate Vec)
                    let total_input_elems = num_leaves * leaf_size;
                    let input_bytes = (total_input_elems * 8) as u64;
                    let input_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("merkle_input"),
                        size: input_bytes,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: true,
                    });
                    {
                        let mut mapped = input_buffer.slice(..).get_mapped_range_mut();
                        for (i, leaf) in leaves.iter().enumerate() {
                            let offset = i * leaf_size * 8;
                            let leaf_bytes: &[u8] = unsafe {
                                std::slice::from_raw_parts(
                                    leaf.as_ptr() as *const u8,
                                    leaf_size * 8,
                                )
                            };
                            mapped[offset..offset + leaf_size * 8].copy_from_slice(leaf_bytes);
                        }
                    }
                    input_buffer.unmap();

                    // Create ping/pong and tree buffers
                    let leaf_hash_bytes = (num_leaves * HASH_SIZE * 8) as u64;
                    let ping_buffer =
                        utils::create_storage_buffer(&ctx.device, leaf_hash_bytes, "merkle_ping");
                    let pong_buffer =
                        utils::create_storage_buffer(&ctx.device, leaf_hash_bytes, "merkle_pong");

                    let cap_len = 1usize << cap_height;
                    let num_digests = 2 * (num_leaves - cap_len);
                    let tree_bytes = (num_digests * HASH_SIZE * 8) as u64;
                    let tree_buffer = utils::create_storage_buffer(
                        &ctx.device,
                        tree_bytes.max(8),
                        "merkle_tree",
                    );

                    let mut encoder =
                        ctx.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("merkle_encoder"),
                            });

                    // Dispatch leaf hashing
                    let use_copy = leaf_size <= HASH_SIZE;
                    let pipeline_name = if use_copy {
                        "copy_row_leaves"
                    } else {
                        "hash_row_leaves"
                    };
                    let leaf_pipeline = ctx.get_pipeline(pipeline_name).unwrap();

                    let leaf_params = MerkleParams {
                        num_leaves: num_leaves as u32,
                        leaf_size: leaf_size as u32,
                        layer_idx: 0,
                        pairs_per_subtree: 0,
                        subtree_digest_len: 0,
                        _pad1: 0,
                        _pad2: 0,
                        _pad3: 0,
                    };
                    let leaf_params_buf = create_uniform_buffer(&ctx.device, &leaf_params);

                    {
                        let bind_group_layout = leaf_pipeline.get_bind_group_layout(0);
                        let bind_group =
                            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("leaf_hash_bg"),
                                layout: &bind_group_layout,
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: input_buffer.as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: ping_buffer.as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 2,
                                        resource: leaf_params_buf.as_entire_binding(),
                                    },
                                ],
                            });

                        let mut pass =
                            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                label: Some("leaf_hash_pass"),
                                timestamp_writes: None,
                            });
                        pass.set_pipeline(leaf_pipeline);
                        pass.set_bind_group(0, Some(&bind_group), &[]);
                        pass.dispatch_workgroups(
                            utils::div_ceil(num_leaves as u32, 256),
                            1,
                            1,
                        );
                        drop(pass);
                    }

                    // Compress layers (ping-pong)
                    let subtree_height = log2_leaves - cap_height;
                    let subtree_digest_len = if subtree_height > 0 {
                        2 * ((1usize << subtree_height) - 1)
                    } else {
                        0
                    };

                    let compress_pipeline = ctx.get_pipeline("compress_nodes").unwrap();
                    let mut current_pairs = num_leaves / 2;
                    let mut current_ping = &ping_buffer;
                    let mut current_pong = &pong_buffer;

                    for layer in 0..subtree_height {
                        let nodes_per_subtree = 1usize << (subtree_height - layer);
                        let pairs_per_subtree = nodes_per_subtree / 2;

                        let layer_params = MerkleParams {
                            num_leaves: current_pairs as u32,
                            leaf_size: 0,
                            layer_idx: layer as u32,
                            pairs_per_subtree: pairs_per_subtree as u32,
                            subtree_digest_len: subtree_digest_len as u32,
                            _pad1: 0,
                            _pad2: 0,
                            _pad3: 0,
                        };
                        let layer_params_buf =
                            create_uniform_buffer(&ctx.device, &layer_params);

                        let bind_group_layout = compress_pipeline.get_bind_group_layout(0);
                        let bind_group =
                            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("compress_bg"),
                                layout: &bind_group_layout,
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: current_ping.as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: current_pong.as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 2,
                                        resource: layer_params_buf.as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 3,
                                        resource: tree_buffer.as_entire_binding(),
                                    },
                                ],
                            });

                        let mut pass =
                            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                label: Some("compress_pass"),
                                timestamp_writes: None,
                            });
                        pass.set_pipeline(compress_pipeline);
                        pass.set_bind_group(0, Some(&bind_group), &[]);
                        pass.dispatch_workgroups(
                            utils::div_ceil(current_pairs as u32, 256),
                            1,
                            1,
                        );
                        drop(pass);

                        std::mem::swap(&mut current_ping, &mut current_pong);
                        current_pairs /= 2;
                    }

                    ctx.queue.submit(Some(encoder.finish()));

                    // Free input_buffer immediately — it's the largest buffer and is no
                    // longer referenced by any pending download commands. The submitted
                    // hash/compress commands keep their own internal reference.
                    input_buffer.destroy();

                    // Read back results
                    let tree_digests: Vec<HashOut<F>> = if num_digests > 0 {
                        let raw = plonky2::maybe_await!(utils::download_field_data(
                            ctx,
                            &tree_buffer,
                            num_digests * HASH_SIZE,
                        ));
                        raw.chunks(HASH_SIZE)
                            .map(|c| HashOut {
                                elements: [c[0], c[1], c[2], c[3]],
                            })
                            .collect()
                    } else {
                        vec![]
                    };

                    let cap_raw = plonky2::maybe_await!(utils::download_field_data(
                        ctx,
                        current_ping,
                        cap_len * HASH_SIZE,
                    ));
                    let cap_hashes: Vec<HashOut<F>> = cap_raw
                        .chunks(HASH_SIZE)
                        .map(|c| HashOut {
                            elements: [c[0], c[1], c[2], c[3]],
                        })
                        .collect();

                    // Free remaining GPU buffers
                    ping_buffer.destroy();
                    pong_buffer.destroy();
                    tree_buffer.destroy();

                    Ok(MerkleTree {
                        leaves,
                        digests: tree_digests,
                        cap: MerkleCap(cap_hashes),
                    })
                })
            }

            $($async_kw)* fn compute_from_coeffs(
                timing: &mut TimingTree,
                polynomials: Vec<PolynomialCoeffs<F>>,
                cap_height: usize,
                rate_bits: usize,
                blinding: bool,
                fft_root_table: Option<&FftRootTable<F>>,
            ) -> anyhow::Result<PolynomialBatch<F, C, D>> {
                let degree = polynomials[0].len();
                let degree_log = log2_strict(degree);
                let lde_degree_log = degree_log + rate_bits;

                // Fall back to CPU for sizes outside GPU twiddle range (12..=22)
                if lde_degree_log < 12 || lde_degree_log > 22 {
                    return plonky2::maybe_await!(
                        <CpuProverCompute as ProverCompute<F, C, D>>::compute_from_coeffs(
                            timing,
                            polynomials,
                            cap_height,
                            rate_bits,
                            blinding,
                            fft_root_table,
                        )
                    );
                }

                const SALT_SIZE: usize = 4;
                let salt_size = if blinding { SALT_SIZE } else { 0 };
                let num_polys = polynomials.len();
                let lde_size = 1 << lde_degree_log;
                let num_cols = num_polys + salt_size;

                let leaves: Vec<Vec<F>> = timed!(
                    timing,
                    "FFT + transpose (WebGPU fused)",
                    {
                        let ctx_guard = crate::context::acquire_context()?;
                        let ctx = &*ctx_guard;

                        let coeff_slices: Vec<&[F]> =
                            polynomials.iter().map(|p| p.coeffs.as_slice()).collect();

                        let mut encoder = ctx.device.create_command_encoder(
                            &wgpu::CommandEncoderDescriptor {
                                label: Some("fused_fft_transpose_encoder"),
                            },
                        );

                        // GPU: upload + to_mont + coset_scale + FFT (results in Montgomery on dst_buffer)
                        let (src_buffer, dst_buffer, shifts_buffer) = fft::encode_coset_fft_batched(
                            ctx,
                            &mut encoder,
                            &coeff_slices,
                            lde_degree_log,
                            F::coset_shift(),
                        )?;

                        // GPU: convert from Montgomery form
                        fft::encode_mont_convert(ctx, &mut encoder, &dst_buffer, lde_size, num_polys, 1);

                        // GPU: transpose + bit_reverse (poly-major → evaluation-major with bit-reversed rows)
                        // Output buffer has num_cols columns per row (leaves room for salt columns)
                        let leaf_buffer_size = (lde_size * num_cols * 8) as u64;
                        let leaf_buffer = utils::create_storage_buffer(
                            &ctx.device,
                            leaf_buffer_size,
                            "transposed_leaves",
                        );
                        fft::encode_transpose_and_bit_reverse(
                            ctx,
                            &mut encoder,
                            &dst_buffer,
                            &leaf_buffer,
                            lde_size,
                            num_polys,
                            num_cols,
                            lde_degree_log,
                        );

                        // Submit all GPU work at once (to_mont + coset_scale + FFT + from_mont + transpose)
                        ctx.queue.submit(Some(encoder.finish()));

                        // Free GPU buffers that are no longer needed BEFORE the download.
                        // The GPU commands are already submitted; the driver keeps internal
                        // references to buffer memory until those commands complete.
                        // Destroying early reduces peak memory by ~160MB, preventing
                        // device loss from OOM on memory-constrained GPUs.
                        src_buffer.destroy();
                        dst_buffer.destroy();
                        shifts_buffer.destroy();

                        // Download transposed leaf data
                        let all_leaf_data = plonky2::maybe_await!(
                            utils::download_field_data(
                                ctx,
                                &leaf_buffer,
                                lde_size * num_cols,
                            )
                        );

                        leaf_buffer.destroy();

                        // Split into rows, adding random salt columns if needed
                        if salt_size == 0 {
                            (0..lde_size)
                                .into_par_iter()
                                .map(|row| {
                                    let start = row * num_cols;
                                    all_leaf_data[start..start + num_cols].to_vec()
                                })
                                .collect()
                        } else {
                            (0..lde_size)
                                .into_par_iter()
                                .map(|row| {
                                    let start = row * num_cols;
                                    let mut row_data = Vec::with_capacity(num_cols);
                                    row_data.extend_from_slice(&all_leaf_data[start..start + num_polys]);
                                    for _ in 0..salt_size {
                                        row_data.push(F::rand());
                                    }
                                    row_data
                                })
                                .collect()
                        }
                    }
                );

                let merkle_tree = plonky2::maybe_await!(
                    <Self as ProverCompute<F, C, D>>::build_merkle_tree(timing, leaves, cap_height)
                )?;

                Ok(PolynomialBatch {
                    polynomials,
                    merkle_tree,
                    degree_log,
                    rate_bits,
                    blinding,
                })
            }

            $($async_kw)* fn compute_from_values(
                timing: &mut TimingTree,
                values: Vec<PolynomialValues<F>>,
                rate_bits: usize,
                blinding: bool,
                cap_height: usize,
                fft_root_table: Option<&FftRootTable<F>>,
            ) -> anyhow::Result<PolynomialBatch<F, C, D>> {
                let degree_log = log2_strict(values[0].len());

                if degree_log < 12 || degree_log + rate_bits > 22 {
                    return plonky2::maybe_await!(
                        <CpuProverCompute as ProverCompute<F, C, D>>::compute_from_values(
                            timing, values, rate_bits, blinding, cap_height, fft_root_table,
                        )
                    );
                }

                let coeffs = timed!(
                    timing,
                    "IFFT (WebGPU)",
                    {
                        let ctx_guard = crate::context::acquire_context()?;
                        let ctx = &*ctx_guard;

                        let value_slices: Vec<&[GoldilocksField]> =
                            values.iter().map(|v| v.values.as_slice()).collect();

                        plonky2::maybe_await!(fft::run_gpu_ifft_batch(ctx, &value_slices))?
                    }
                );

                plonky2::maybe_await!(Self::compute_from_coeffs(
                    timing, coeffs, cap_height, rate_bits, blinding, fft_root_table
                ))
            }

            $($async_kw)* fn transpose_and_compute_from_coeffs(
                timing: &mut TimingTree,
                pre_transposed_quotient_polys: Vec<Vec<F>>,
                quotient_degree: usize,
                degree: usize,
                rate_bits: usize,
                blinding: bool,
                cap_height: usize,
                fft_root_table: Option<&FftRootTable<F>>,
            ) -> anyhow::Result<PolynomialBatch<F, C, D>> {
                let ifft_log_n = log2_strict(pre_transposed_quotient_polys.len());
                let degree_log = log2_strict(degree);

                if ifft_log_n < 12 || ifft_log_n > 22 || degree_log + rate_bits > 22 {
                    return plonky2::maybe_await!(
                        <CpuProverCompute as ProverCompute<F, C, D>>::transpose_and_compute_from_coeffs(
                            timing,
                            pre_transposed_quotient_polys,
                            quotient_degree,
                            degree,
                            rate_bits,
                            blinding,
                            cap_height,
                            fft_root_table,
                        )
                    );
                }

                let quotient_polys: Vec<PolynomialCoeffs<F>> = timed!(
                    timing,
                    "coset IFFT quotient polys (WebGPU)",
                    {
                        let ctx_guard = crate::context::acquire_context()?;
                        let ctx = &*ctx_guard;

                        let transposed = plonky2::util::transpose(&pre_transposed_quotient_polys);
                        let value_slices: Vec<&[GoldilocksField]> =
                            transposed.iter().map(|v| v.as_slice()).collect();

                        let mut coeffs_batch = plonky2::maybe_await!(
                            fft::run_gpu_ifft_batch(ctx, &value_slices)
                        )?;

                        // Apply coset shift inverse
                        let n = coeffs_batch[0].len();
                        let shift_inv = F::coset_shift().inverse();
                        let powers_of_shift_inv: Vec<_> = shift_inv.powers().take(n).collect();
                        coeffs_batch.par_iter_mut().for_each(|coeffs| {
                            coeffs
                                .coeffs
                                .iter_mut()
                                .zip(powers_of_shift_inv.iter())
                                .for_each(|(c, s)| *c *= *s);
                        });

                        coeffs_batch
                    }
                );

                let all_quotient_poly_chunks: Vec<PolynomialCoeffs<F>> = timed!(
                    timing,
                    "split up quotient polys",
                    quotient_polys
                        .into_par_iter()
                        .flat_map(|mut quotient_poly| {
                            quotient_poly.trim_to_len(quotient_degree).expect(
                                "Quotient has failed, the vanishing polynomial is not divisible by Z_H",
                            );
                            quotient_poly.chunks(degree)
                        })
                        .collect()
                );

                plonky2::maybe_await!(Self::compute_from_coeffs(
                    timing,
                    all_quotient_poly_chunks,
                    cap_height,
                    rate_bits,
                    blinding,
                    fft_root_table,
                ))
            }
        }
    };
}

#[cfg(not(feature = "async_prover"))]
impl_webgpu_prover_compute!();

#[cfg(feature = "async_prover")]
impl_webgpu_prover_compute!(async);

#[cfg(test)]
mod tests {
    use super::*;
    use plonky2::field::types::Sample;

    #[test]
    fn test_fft_consistency() {
        let ctx_guard = crate::context::acquire_context().expect("Failed to get WebGPU context");
        let ctx = &*ctx_guard;

        for log_n in 12..=16 {
            let n = 1 << log_n;
            let coeffs = PolynomialCoeffs::new(GoldilocksField::rand_vec(n));
            let shift = GoldilocksField::coset_shift();

            let gpu_coset_fft = fft::run_gpu_coset_fft_batch(
                ctx,
                &[coeffs.coeffs.as_slice()],
                log_n,
                shift,
            )
            .unwrap();

            let cpu_coset_fft = coeffs.coset_fft(shift).values;
            assert_eq!(
                gpu_coset_fft[0], cpu_coset_fft,
                "Coset FFT mismatch for log_n = {}",
                log_n
            );
        }
    }

    #[test]
    fn test_ifft_consistency() {
        let ctx_guard = crate::context::acquire_context().expect("Failed to get WebGPU context");
        let ctx = &*ctx_guard;

        for log_n in 12..=16 {
            let n = 1 << log_n;
            let coeffs = PolynomialCoeffs::new(GoldilocksField::rand_vec(n));
            let shift = GoldilocksField::coset_shift();
            let cpu_coset_fft = coeffs.coset_fft(shift).values;

            let values = PolynomialValues::new(cpu_coset_fft);
            let gpu_ifft =
                fft::run_gpu_ifft_batch(ctx, &[values.values.as_slice()]).unwrap();
            let cpu_ifft = values.ifft();

            assert_eq!(
                gpu_ifft[0].coeffs, cpu_ifft.coeffs,
                "IFFT mismatch for log_n = {}",
                log_n
            );
        }
    }

    #[test]
    fn test_mont_roundtrip() {
        for _ in 0..1000 {
            let val = GoldilocksField::rand();
            let mont = utils::to_mont(val.0);
            let back = utils::from_mont(mont);
            assert_eq!(val.0, back, "Montgomery round-trip failed for {}", val.0);
        }
    }

    #[test]
    fn test_gpu_mont_convert_consistency() {
        // Test that GPU-side Montgomery conversion matches CPU-side
        let ctx_guard = crate::context::acquire_context().expect("Failed to get WebGPU context");
        let ctx = &*ctx_guard;

        let n = 4096;
        let data: Vec<GoldilocksField> = GoldilocksField::rand_vec(n);

        // Upload raw data
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mont_test"),
            size: (n * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u8, n * 8)
            };
            buffer.slice(..).get_mapped_range_mut().copy_from_slice(bytes);
        }
        buffer.unmap();

        // GPU: to_mont then from_mont (round-trip)
        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mont_test_encoder"),
        });
        fft::encode_mont_convert(ctx, &mut encoder, &buffer, n, 1, 0); // to_mont
        fft::encode_mont_convert(ctx, &mut encoder, &buffer, n, 1, 1); // from_mont
        ctx.queue.submit(Some(encoder.finish()));

        let result = utils::download_field_data(ctx, &buffer, n);
        assert_eq!(data, result, "GPU Montgomery round-trip failed");
    }

    #[test]
    fn test_gpu_transpose_consistency() {
        // Test that GPU transpose + bit_reverse matches CPU transpose + reverse_index_bits_in_place
        let ctx_guard = crate::context::acquire_context().expect("Failed to get WebGPU context");
        let ctx = &*ctx_guard;

        let log_lde_size = 12;
        let lde_size = 1 << log_lde_size;
        let num_polys = 8;

        // Generate random polynomial data
        let poly_data: Vec<Vec<GoldilocksField>> = (0..num_polys)
            .map(|_| GoldilocksField::rand_vec(lde_size))
            .collect();

        // CPU reference: transpose + bit_reverse
        let lde_refs: Vec<Vec<F>> = poly_data.clone();
        let mut cpu_leaves = plonky2::util::transpose(&lde_refs);
        plonky2_util::reverse_index_bits_in_place(&mut cpu_leaves);

        // GPU: upload poly-major data, transpose + bit_reverse
        let src_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("transpose_test_src"),
            size: (num_polys * lde_size * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let mut mapped = src_buffer.slice(..).get_mapped_range_mut();
            for (i, poly) in poly_data.iter().enumerate() {
                let offset = i * lde_size * 8;
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(poly.as_ptr() as *const u8, lde_size * 8)
                };
                mapped[offset..offset + lde_size * 8].copy_from_slice(bytes);
            }
        }
        src_buffer.unmap();

        let dst_buffer = utils::create_storage_buffer(
            &ctx.device,
            (lde_size * num_polys * 8) as u64,
            "transpose_test_dst",
        );

        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("transpose_test_encoder"),
        });
        fft::encode_transpose_and_bit_reverse(
            ctx,
            &mut encoder,
            &src_buffer,
            &dst_buffer,
            lde_size,
            num_polys,
            num_polys,
            log_lde_size,
        );
        ctx.queue.submit(Some(encoder.finish()));

        let gpu_data = utils::download_field_data(
            ctx,
            &dst_buffer,
            lde_size * num_polys,
        );

        // Compare row by row
        for row in 0..lde_size {
            let start = row * num_polys;
            let gpu_row = &gpu_data[start..start + num_polys];
            assert_eq!(
                gpu_row, cpu_leaves[row].as_slice(),
                "Transpose mismatch at row {}",
                row
            );
        }
    }

    #[test]
    fn test_merkle_tree_consistency() {
        for log2_leaves in [4, 8, 12] {
            let n_leaves = 1usize << log2_leaves;
            for &leaf_size in &[4, 8, 20] {
                let cap_height = 4.min(log2_leaves);
                let leaves: Vec<Vec<F>> =
                    (0..n_leaves).map(|_| F::rand_vec(leaf_size)).collect();

                let cpu_tree =
                    MerkleTree::<F, PoseidonHash>::new(leaves.clone(), cap_height);

                let mut timing = TimingTree::default();
                let gpu_tree = <WebGpuProverCompute as ProverCompute<F, C, D>>::build_merkle_tree(
                    &mut timing,
                    leaves.clone(),
                    cap_height,
                )
                .expect("GPU Merkle tree failed");

                assert_eq!(
                    cpu_tree.cap, gpu_tree.cap,
                    "Merkle cap mismatch for log2_leaves={} leaf_size={}",
                    log2_leaves, leaf_size,
                );
                assert_eq!(
                    cpu_tree.digests, gpu_tree.digests,
                    "Merkle digests mismatch for log2_leaves={} leaf_size={}",
                    log2_leaves, leaf_size,
                );
            }
        }
    }
}
