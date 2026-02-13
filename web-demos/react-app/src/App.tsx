import React, { useState, useRef, useCallback, useEffect } from 'react';

interface CircuitInitResult {
  circuit_build_ms: number;
  recursive_circuit_build_ms: number;
  total_ms: number;
  success: boolean;
  error?: string;
}

interface ProofResult {
  inner_proof_1_ms: number;
  inner_proof_2_ms: number;
  recursive_proof_ms: number;
  verification_ms: number;
  total_ms: number;
  success: boolean;
  error?: string;
}

type RunState = 'idle' | 'running' | 'done' | 'error';

function formatMs(ms: number): string {
  if (ms === 0) return '-';
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

function speedup(cpu: number, gpu: number): string {
  if (cpu === 0 || gpu === 0) return '-';
  const ratio = cpu / gpu;
  if (ratio >= 1) return `${ratio.toFixed(2)}x faster`;
  return `${(1 / ratio).toFixed(2)}x slower`;
}

type PhaseSource = 'init' | 'proof';

const PHASES: { key: string; label: string; source: PhaseSource }[] = [
  { key: 'circuit_build_ms', label: 'Inner Circuit Build', source: 'init' },
  { key: 'recursive_circuit_build_ms', label: 'Recursive Circuit Build', source: 'init' },
  { key: 'inner_proof_1_ms', label: 'Inner Proof #1', source: 'proof' },
  { key: 'inner_proof_2_ms', label: 'Inner Proof #2', source: 'proof' },
  { key: 'recursive_proof_ms', label: 'Recursive Proof', source: 'proof' },
  { key: 'verification_ms', label: 'Verification (3+9 proofs)', source: 'proof' },
];

function getPhaseValue(
  source: PhaseSource,
  key: string,
  initResult: CircuitInitResult | null,
  proofResult: ProofResult | null,
): number | null {
  const result = source === 'init' ? initResult : proofResult;
  if (!result) return null;
  return (result as any)[key] as number;
}

function totalMs(
  initResult: CircuitInitResult | null,
  proofResult: ProofResult | null,
): number {
  return (initResult?.total_ms ?? 0) + (proofResult?.total_ms ?? 0);
}

/** Detect iOS/iPadOS via user-agent heuristics. */
function detectMobilePlatform(): boolean {
  const ua = navigator.userAgent || '';
  // iPhone, iPad (old UA), iPod
  if (/iPhone|iPad|iPod/.test(ua)) return true;
  // iPadOS 13+ reports as Mac but with touch support
  if (/Macintosh/.test(ua) && navigator.maxTouchPoints > 1) return true;
  // Android mobile
  if (/Android/.test(ua) && /Mobile/.test(ua)) return true;
  return false;
}

export default function App() {
  const [cpuState, setCpuState] = useState<RunState>('idle');
  const [gpuState, setGpuState] = useState<RunState>('idle');
  const [cpuInitResult, setCpuInitResult] = useState<CircuitInitResult | null>(null);
  const [cpuProofResult, setCpuProofResult] = useState<ProofResult | null>(null);
  const [gpuInitResult, setGpuInitResult] = useState<CircuitInitResult | null>(null);
  const [gpuProofResult, setGpuProofResult] = useState<ProofResult | null>(null);
  const [cpuStatus, setCpuStatus] = useState('');
  const [gpuStatus, setGpuStatus] = useState('');
  const [hasWebGPU, setHasWebGPU] = useState<boolean | null>(null);
  const [mobileMode, setMobileMode] = useState<boolean>(detectMobilePlatform);

  const cpuWorkerRef = useRef<Worker | null>(null);
  const gpuWorkerRef = useRef<Worker | null>(null);

  // Check WebGPU support on mount
  useEffect(() => {
    const check = async () => {
      if ('gpu' in navigator) {
        try {
          const adapter = await (navigator as any).gpu.requestAdapter();
          setHasWebGPU(adapter !== null);
        } catch {
          setHasWebGPU(false);
        }
      } else {
        setHasWebGPU(false);
      }
    };
    check();
    return () => {
      cpuWorkerRef.current?.terminate();
      gpuWorkerRef.current?.terminate();
    };
  }, []);

  const getOrCreateCpuWorker = useCallback(() => {
    if (!cpuWorkerRef.current) {
      const worker = new Worker(
        new URL('./workers/cpu-worker.ts', import.meta.url),
        { type: 'module' }
      );
      worker.onmessage = (e: MessageEvent) => {
        if (e.data.type === 'status') {
          setCpuStatus(e.data.status);
        } else if (e.data.type === 'init_result') {
          const result = e.data.result as CircuitInitResult;
          setCpuInitResult(result);
          if (!result.success) {
            setCpuState('error');
            setCpuStatus(`Error: ${result.error}`);
          }
        } else if (e.data.type === 'proof_result') {
          const result = e.data.result as ProofResult;
          setCpuProofResult(result);
          setCpuState(result.success ? 'done' : 'error');
          setCpuStatus(result.success ? 'Complete' : `Error: ${result.error}`);
        } else if (e.data.type === 'error') {
          setCpuState('error');
          setCpuStatus(e.data.error);
        }
      };
      worker.onerror = (err) => {
        setCpuState('error');
        setCpuStatus(`Worker error: ${err.message}`);
      };
      cpuWorkerRef.current = worker;
    }
    return cpuWorkerRef.current;
  }, []);

  const getOrCreateGpuWorker = useCallback(() => {
    if (!gpuWorkerRef.current) {
      const worker = new Worker(
        new URL('./workers/webgpu-worker.ts', import.meta.url),
        { type: 'module' }
      );
      worker.onmessage = (e: MessageEvent) => {
        if (e.data.type === 'status') {
          setGpuStatus(e.data.status);
        } else if (e.data.type === 'init_result') {
          const result = e.data.result as CircuitInitResult;
          setGpuInitResult(result);
          if (!result.success) {
            setGpuState('error');
            setGpuStatus(`Error: ${result.error}`);
          }
        } else if (e.data.type === 'proof_result') {
          const result = e.data.result as ProofResult;
          setGpuProofResult(result);
          setGpuState(result.success ? 'done' : 'error');
          setGpuStatus(result.success ? 'Complete' : `Error: ${result.error}`);
        } else if (e.data.type === 'error') {
          setGpuState('error');
          setGpuStatus(e.data.error);
        }
      };
      worker.onerror = (err) => {
        setGpuState('error');
        setGpuStatus(`Worker error: ${err.message}`);
      };
      gpuWorkerRef.current = worker;
    }
    return gpuWorkerRef.current;
  }, []);

  const runCpu = useCallback(() => {
    setCpuState('running');
    setCpuInitResult(null);
    setCpuProofResult(null);
    setCpuStatus('Starting...');
    getOrCreateCpuWorker().postMessage({ type: 'run' });
  }, [getOrCreateCpuWorker]);

  const rerunCpuProofs = useCallback(() => {
    setCpuState('running');
    setCpuProofResult(null);
    setCpuStatus('Re-running proofs...');
    getOrCreateCpuWorker().postMessage({ type: 'prove' });
  }, [getOrCreateCpuWorker]);

  const runGpu = useCallback(() => {
    setGpuState('running');
    setGpuInitResult(null);
    setGpuProofResult(null);
    setGpuStatus('Starting...');
    getOrCreateGpuWorker().postMessage({ type: 'run', mobile: mobileMode });
  }, [getOrCreateGpuWorker, mobileMode]);

  const rerunGpuProofs = useCallback(() => {
    setGpuState('running');
    setGpuProofResult(null);
    setGpuStatus('Re-running proofs...');
    getOrCreateGpuWorker().postMessage({ type: 'prove' });
  }, [getOrCreateGpuWorker]);

  const runBoth = useCallback(() => {
    runCpu();
    if (hasWebGPU) runGpu();
  }, [runCpu, runGpu, hasWebGPU]);

  const cpuInitDone = cpuInitResult?.success === true;
  const gpuInitDone = gpuInitResult?.success === true;

  return (
    <div style={{ fontFamily: 'monospace', maxWidth: 900, margin: '0 auto', padding: 20 }}>
      <h1 style={{ fontSize: 24 }}>Plonky2 WASM Benchmark</h1>
      <p style={{ color: '#666', marginBottom: 8 }}>
        CPU (PoseidonGoldilocksConfig) vs WebGPU (PoseidonGoldilocksWebGpuConfig)
      </p>
      <p style={{ color: '#666', fontSize: 12 }}>
        Circuit: DummyPsyTypeCCircuit + DummyPsyTypeCRecursiveVerifierCircuit (psy_bench_recursion)
      </p>

      <div style={{ marginBottom: 16, display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
        <span style={{
          display: 'inline-block',
          padding: '2px 8px',
          borderRadius: 4,
          fontSize: 12,
          background: hasWebGPU === null ? '#888' : hasWebGPU ? '#2a7' : '#c44',
          color: '#fff',
        }}>
          WebGPU: {hasWebGPU === null ? 'checking...' : hasWebGPU ? 'supported' : 'not available'}
        </span>
        <label style={{ fontSize: 12, display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={mobileMode}
            onChange={(e) => setMobileMode(e.target.checked)}
            disabled={gpuState === 'running'}
          />
          Mobile / Low Memory Mode
          <span style={{ color: '#999' }}>
            (chunked GPU downloads, auto-detected: {detectMobilePlatform() ? 'yes' : 'no'})
          </span>
        </label>
      </div>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        <button
          onClick={runCpu}
          disabled={cpuState === 'running'}
          style={btnStyle}
        >
          {cpuState === 'running' ? 'Running CPU...' : 'Run CPU'}
        </button>
        <button
          onClick={rerunCpuProofs}
          disabled={cpuState === 'running' || !cpuInitDone}
          style={{ ...btnStyle, background: cpuInitDone ? '#555' : '#999' }}
        >
          Re-run CPU Proofs
        </button>
        <button
          onClick={runGpu}
          disabled={gpuState === 'running' || !hasWebGPU}
          style={btnStyle}
        >
          {gpuState === 'running' ? 'Running WebGPU...' : 'Run WebGPU'}
        </button>
        <button
          onClick={rerunGpuProofs}
          disabled={gpuState === 'running' || !gpuInitDone}
          style={{ ...btnStyle, background: gpuInitDone ? '#555' : '#999' }}
        >
          Re-run GPU Proofs
        </button>
        <button
          onClick={runBoth}
          disabled={cpuState === 'running' || gpuState === 'running'}
          style={{ ...btnStyle, background: '#2a6' }}
        >
          Run Both
        </button>
      </div>

      {/* Status */}
      <div style={{ display: 'flex', gap: 20, marginBottom: 20 }}>
        <div style={{ flex: 1 }}>
          <strong>CPU Status:</strong>{' '}
          <span style={{ color: cpuState === 'error' ? '#c44' : cpuState === 'done' ? '#2a7' : '#888' }}>
            {cpuStatus || 'idle'}
          </span>
        </div>
        <div style={{ flex: 1 }}>
          <strong>WebGPU Status:</strong>{' '}
          <span style={{ color: gpuState === 'error' ? '#c44' : gpuState === 'done' ? '#2a7' : '#888' }}>
            {gpuStatus || 'idle'}
          </span>
        </div>
      </div>

      {/* Results table */}
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 14 }}>
        <thead>
          <tr style={{ borderBottom: '2px solid #333' }}>
            <th style={thStyle}>Phase</th>
            <th style={thStyle}>CPU (WASM)</th>
            <th style={thStyle}>WebGPU (WASM)</th>
            <th style={thStyle}>Comparison</th>
          </tr>
        </thead>
        <tbody>
          {/* Init phases header */}
          <tr>
            <td colSpan={4} style={{ ...tdStyle, fontWeight: 'bold', background: '#f5f5f5', fontSize: 12, color: '#666' }}>
              Phase 1: Circuit Building {cpuInitDone || gpuInitDone ? '(cached)' : ''}
            </td>
          </tr>
          {PHASES.filter(p => p.source === 'init').map(({ key, label, source }) => {
            const cpuVal = getPhaseValue(source, key, cpuInitResult, cpuProofResult);
            const gpuVal = getPhaseValue(source, key, gpuInitResult, gpuProofResult);
            return (
              <tr key={key} style={{ borderBottom: '1px solid #ddd' }}>
                <td style={tdStyle}>{label}</td>
                <td style={{ ...tdStyle, textAlign: 'right' }}>
                  {cpuVal !== null ? formatMs(cpuVal) : '-'}
                </td>
                <td style={{ ...tdStyle, textAlign: 'right' }}>
                  {gpuVal !== null ? formatMs(gpuVal) : '-'}
                </td>
                <td style={{ ...tdStyle, textAlign: 'right' }}>
                  {cpuVal !== null && gpuVal !== null ? speedup(cpuVal, gpuVal) : '-'}
                </td>
              </tr>
            );
          })}
          {/* Proof phases header */}
          <tr>
            <td colSpan={4} style={{ ...tdStyle, fontWeight: 'bold', background: '#f5f5f5', fontSize: 12, color: '#666' }}>
              Phase 2: Proving &amp; Verification
            </td>
          </tr>
          {PHASES.filter(p => p.source === 'proof').map(({ key, label, source }) => {
            const cpuVal = getPhaseValue(source, key, cpuInitResult, cpuProofResult);
            const gpuVal = getPhaseValue(source, key, gpuInitResult, gpuProofResult);
            return (
              <tr key={key} style={{ borderBottom: '1px solid #ddd' }}>
                <td style={tdStyle}>{label}</td>
                <td style={{ ...tdStyle, textAlign: 'right' }}>
                  {cpuVal !== null ? formatMs(cpuVal) : '-'}
                </td>
                <td style={{ ...tdStyle, textAlign: 'right' }}>
                  {gpuVal !== null ? formatMs(gpuVal) : '-'}
                </td>
                <td style={{ ...tdStyle, textAlign: 'right' }}>
                  {cpuVal !== null && gpuVal !== null ? speedup(cpuVal, gpuVal) : '-'}
                </td>
              </tr>
            );
          })}
          {/* Total row */}
          <tr style={{ borderTop: '2px solid #333' }}>
            <td style={{ ...tdStyle, fontWeight: 'bold' }}>Total</td>
            <td style={{ ...tdStyle, textAlign: 'right', fontWeight: 'bold' }}>
              {cpuInitResult || cpuProofResult
                ? formatMs(totalMs(cpuInitResult, cpuProofResult))
                : '-'}
            </td>
            <td style={{ ...tdStyle, textAlign: 'right', fontWeight: 'bold' }}>
              {gpuInitResult || gpuProofResult
                ? formatMs(totalMs(gpuInitResult, gpuProofResult))
                : '-'}
            </td>
            <td style={{ ...tdStyle, textAlign: 'right', fontWeight: 'bold' }}>
              {(cpuInitResult || cpuProofResult) && (gpuInitResult || gpuProofResult)
                ? speedup(
                    totalMs(cpuInitResult, cpuProofResult),
                    totalMs(gpuInitResult, gpuProofResult),
                  )
                : '-'}
            </td>
          </tr>
        </tbody>
      </table>

      {/* Errors */}
      {(cpuInitResult?.error || cpuProofResult?.error) && (
        <div style={{ color: '#c44', marginTop: 12, whiteSpace: 'pre-wrap', fontSize: 12 }}>
          <strong>CPU Error:</strong> {cpuInitResult?.error || cpuProofResult?.error}
        </div>
      )}
      {(gpuInitResult?.error || gpuProofResult?.error) && (
        <div style={{ color: '#c44', marginTop: 12, whiteSpace: 'pre-wrap', fontSize: 12 }}>
          <strong>WebGPU Error:</strong> {gpuInitResult?.error || gpuProofResult?.error}
        </div>
      )}

      <div style={{ marginTop: 30, color: '#999', fontSize: 11 }}>
        <p>Notes:</p>
        <ul>
          <li>Both benchmarks run the same psy_bench_recursion circuit from plonky2</li>
          <li>CPU uses PoseidonGoldilocksConfig (standard single-threaded WASM)</li>
          <li>WebGPU uses PoseidonGoldilocksWebGpuConfig (GPU-accelerated FFT + Merkle trees)</li>
          <li>Circuit build time is identical (CPU-only operation in both cases)</li>
          <li>WebGPU acceleration applies to proving (FFT, Merkle tree construction)</li>
          <li>Verification is always CPU (no GPU needed for verification)</li>
          <li>After initial run, use "Re-run Proofs" to re-prove with cached circuits</li>
          <li>WebGPU requires Chrome 113+, Edge 113+, or Safari 26+ (iOS/macOS)</li>
          <li>Mobile mode enables chunked GPU downloads (16 MiB) to prevent OOM on iOS Safari.
              Auto-enabled when the adapter reports max_buffer_size &le; 256 MiB</li>
        </ul>
      </div>
    </div>
  );
}

const btnStyle: React.CSSProperties = {
  padding: '8px 16px',
  fontSize: 14,
  fontFamily: 'monospace',
  border: '1px solid #333',
  borderRadius: 4,
  background: '#333',
  color: '#fff',
  cursor: 'pointer',
};

const thStyle: React.CSSProperties = {
  textAlign: 'left',
  padding: '8px 12px',
};

const tdStyle: React.CSSProperties = {
  padding: '6px 12px',
};
