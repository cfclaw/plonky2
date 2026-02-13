import init, {
  init as initPanic,
  init_gpu,
  configure_gpu_for_mobile,
  get_gpu_max_buffer_size,
  get_gpu_download_chunk_size,
  build_circuits_webgpu,
  run_proofs_webgpu,
} from 'plonky2-wasm-webgpu';

let wasmInitialized = false;

self.onmessage = async (e: MessageEvent) => {
  try {
    if (e.data.type === 'init' || e.data.type === 'run') {
      if (!wasmInitialized) {
        self.postMessage({ type: 'status', status: 'Initializing WASM module...' });
        await init();
        initPanic();
        self.postMessage({ type: 'status', status: 'Initializing WebGPU device...' });
        await init_gpu();

        // Apply mobile configuration if requested or auto-detected.
        // The WASM layer auto-enables chunking when the adapter reports
        // max_buffer_size ≤ 256 MiB. The explicit flag from the UI is an
        // additional manual override for devices that report higher limits
        // but still suffer from memory pressure (e.g. newer iPhones).
        const mobileMode = e.data.mobile === true;
        const maxBuf = get_gpu_max_buffer_size();
        const autoChunk = get_gpu_download_chunk_size();

        if (mobileMode && autoChunk === BigInt(0)) {
          configure_gpu_for_mobile();
          self.postMessage({
            type: 'status',
            status: `Mobile mode enabled (manual). Adapter max_buffer_size: ${maxBuf}`,
          });
        } else if (autoChunk > BigInt(0)) {
          self.postMessage({
            type: 'status',
            status: `Mobile mode auto-enabled (chunk=${autoChunk}). Adapter max_buffer_size: ${maxBuf}`,
          });
        }

        wasmInitialized = true;
      }

      self.postMessage({ type: 'status', status: 'Building circuits...' });
      const initResult = build_circuits_webgpu();
      self.postMessage({ type: 'init_result', result: initResult });

      if (!initResult.success) return;

      if (e.data.type === 'run') {
        self.postMessage({ type: 'status', status: 'Generating and verifying proofs...' });
        const proofResult = await run_proofs_webgpu();
        self.postMessage({ type: 'proof_result', result: proofResult });
      }
    } else if (e.data.type === 'prove') {
      self.postMessage({ type: 'status', status: 'Generating and verifying proofs...' });
      const proofResult = await run_proofs_webgpu();
      self.postMessage({ type: 'proof_result', result: proofResult });
    }
  } catch (err: any) {
    self.postMessage({
      type: 'error',
      error: `Worker error: ${err.message || err}`,
    });
  }
};
