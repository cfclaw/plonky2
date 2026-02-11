use anyhow::{anyhow, Result};
use lazy_static::lazy_static;
use parking_lot::Mutex;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::ops::Square;
use plonky2::field::types::Field;
use std::collections::HashMap;

use crate::utils;

// Shader sources: goldilocks preamble + kernel-specific code, assembled by build.rs
const FFT_COSET_SCALE_WGSL: &str = include_str!(concat!(env!("OUT_DIR"), "/fft_coset_scale.wgsl"));
const FFT_DIT_PASS_WGSL: &str = include_str!(concat!(env!("OUT_DIR"), "/fft_dit_pass.wgsl"));
const BIT_REVERSE_COPY_WGSL: &str = include_str!(concat!(env!("OUT_DIR"), "/bit_reverse_copy.wgsl"));
const IFFT_REORDER_WGSL: &str = include_str!(concat!(env!("OUT_DIR"), "/ifft_reorder.wgsl"));
const MERKLE_WGSL: &str = include_str!(concat!(env!("OUT_DIR"), "/merkle.wgsl"));

const MAX_DEGREE_LOG: usize = 18;
const MAX_RATE_BITS: usize = 4;

pub struct WebGpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub pipelines: HashMap<String, wgpu::ComputePipeline>,
    /// Pre-computed twiddle factors in Montgomery form, keyed by log_n.
    pub twiddle_buffers: HashMap<usize, wgpu::Buffer>,
}

impl WebGpuContext {
    pub fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| anyhow!("No suitable GPU adapter found"))?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("plonky2_webgpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    max_storage_buffer_binding_size: 1 << 30, // 1 GB
                    max_buffer_size: 1 << 30,
                    max_compute_workgroups_per_dimension: 65535,
                    ..wgpu::Limits::default()
                },
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|e| anyhow!("Failed to create WebGPU device: {}", e))?;

        let mut ctx = Self {
            device,
            queue,
            pipelines: HashMap::new(),
            twiddle_buffers: HashMap::new(),
        };

        // Compile FFT pipelines
        ctx.compile_pipeline("coset_scale_and_bit_reverse_batched", FFT_COSET_SCALE_WGSL)?;
        ctx.compile_pipeline("fft_dit_pass_batched", FFT_DIT_PASS_WGSL)?;
        ctx.compile_pipeline("bit_reverse_copy_batched", BIT_REVERSE_COPY_WGSL)?;
        ctx.compile_pipeline("ifft_reorder_and_scale_batched", IFFT_REORDER_WGSL)?;

        // Compile Merkle pipelines
        ctx.compile_pipeline("copy_row_leaves", MERKLE_WGSL)?;
        ctx.compile_pipeline("hash_row_leaves", MERKLE_WGSL)?;
        ctx.compile_pipeline("compress_nodes", MERKLE_WGSL)?;

        // Pre-compute twiddle factors in Montgomery form
        for log_n in 12..=(MAX_DEGREE_LOG + MAX_RATE_BITS) {
            let fwd_generator = GoldilocksField::primitive_root_of_unity(log_n);
            let root_table = fft_root_table_from_generator(fwd_generator, log_n);
            let twiddles_flat: Vec<u64> = root_table
                .into_iter()
                .flatten()
                .map(|f| utils::to_mont(f.0))
                .collect();

            let twiddle_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    twiddles_flat.as_ptr() as *const u8,
                    twiddles_flat.len() * std::mem::size_of::<u64>(),
                )
            };

            let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("twiddles_{}", log_n)),
                size: twiddle_bytes.len() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: true,
            });
            buffer.slice(..).get_mapped_range_mut().copy_from_slice(twiddle_bytes);
            buffer.unmap();

            ctx.twiddle_buffers.insert(log_n, buffer);
        }

        Ok(ctx)
    }

    fn compile_pipeline(&mut self, entry_point: &str, shader_source: &str) -> Result<()> {
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(entry_point),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: None, // auto-layout from shader
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            });

        self.pipelines.insert(entry_point.to_string(), pipeline);
        Ok(())
    }

    pub fn get_pipeline(&self, name: &str) -> Option<&wgpu::ComputePipeline> {
        self.pipelines.get(name)
    }
}

/// Generate twiddle table from a root of unity generator.
/// Matches the Metal implementation's fft_root_table_from_generator.
fn fft_root_table_from_generator(
    mut base: GoldilocksField,
    max_log_n: usize,
) -> Vec<Vec<GoldilocksField>> {
    let mut bases = Vec::with_capacity(max_log_n);
    bases.push(base);
    for _ in 1..max_log_n {
        base = base.square();
        bases.push(base);
    }
    bases.reverse();

    (1..=max_log_n)
        .map(|log_m| {
            let m = 1 << log_m;
            let base = bases[log_m - 1];
            base.powers().take(m / 2).collect()
        })
        .collect()
}

lazy_static! {
    pub static ref WEBGPU_CONTEXT: Mutex<Option<WebGpuContext>> = {
        match WebGpuContext::new() {
            Ok(ctx) => Mutex::new(Some(ctx)),
            Err(e) => {
                println!("Failed to initialize WebGPU backend: {:?}", e);
                Mutex::new(None)
            }
        }
    };
}

pub fn get_context<'a>() -> Option<parking_lot::MutexGuard<'a, Option<WebGpuContext>>> {
    WEBGPU_CONTEXT.try_lock()
}
