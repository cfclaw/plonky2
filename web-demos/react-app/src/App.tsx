import React, { useState, useRef, useCallback } from 'react';

interface BenchmarkResult {
  circuit_build_ms: number;
  inner_proof_1_ms: number;
  inner_proof_2_ms: number;
  recursive_circuit_build_ms: number;
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

const PHASES = [
  { key: 'circuit_build_ms', label: 'Inner Circuit Build' },
  { key: 'inner_proof_1_ms', label: 'Inner Proof #1' },
  { key: 'inner_proof_2_ms', label: 'Inner Proof #2' },
  { key: 'recursive_circuit_build_ms', label: 'Recursive Circuit Build' },
  { key: 'recursive_proof_ms', label: 'Recursive Proof' },
  { key: 'verification_ms', label: 'Verification (3+9 proofs)' },
  { key: 'total_ms', label: 'Total' },
] as const;

export default function App() {
  const [cpuState, setCpuState] = useState<RunState>('idle');
  const [gpuState, setGpuState] = useState<RunState>('idle');
  const [cpuResult, setCpuResult] = useState<BenchmarkResult | null>(null);
  const [gpuResult, setGpuResult] = useState<BenchmarkResult | null>(null);
  const [cpuStatus, setCpuStatus] = useState('');
  const [gpuStatus, setGpuStatus] = useState('');
  const [hasWebGPU, setHasWebGPU] = useState<boolean | null>(null);

  const cpuWorkerRef = useRef<Worker | null>(null);
  const gpuWorkerRef = useRef<Worker | null>(null);

  // Check WebGPU support on mount
  React.useEffect(() => {
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
  }, []);

  const runCpu = useCallback(() => {
    setCpuState('running');
    setCpuResult(null);
    setCpuStatus('Starting...');

    const worker = new Worker(
      new URL('./workers/cpu-worker.ts', import.meta.url),
      { type: 'module' }
    );
    cpuWorkerRef.current = worker;

    worker.onmessage = (e: MessageEvent) => {
      if (e.data.type === 'status') {
        setCpuStatus(e.data.status);
      } else if (e.data.type === 'result') {
        const result = e.data.result as BenchmarkResult;
        setCpuResult(result);
        setCpuState(result.success ? 'done' : 'error');
        setCpuStatus(result.success ? 'Complete' : `Error: ${result.error}`);
        worker.terminate();
      }
    };

    worker.onerror = (err) => {
      setCpuState('error');
      setCpuStatus(`Worker error: ${err.message}`);
      worker.terminate();
    };

    worker.postMessage({ type: 'run' });
  }, []);

  const runGpu = useCallback(() => {
    setGpuState('running');
    setGpuResult(null);
    setGpuStatus('Starting...');

    const worker = new Worker(
      new URL('./workers/webgpu-worker.ts', import.meta.url),
      { type: 'module' }
    );
    gpuWorkerRef.current = worker;

    worker.onmessage = (e: MessageEvent) => {
      if (e.data.type === 'status') {
        setGpuStatus(e.data.status);
      } else if (e.data.type === 'result') {
        const result = e.data.result as BenchmarkResult;
        setGpuResult(result);
        setGpuState(result.success ? 'done' : 'error');
        setGpuStatus(result.success ? 'Complete' : `Error: ${result.error}`);
        worker.terminate();
      }
    };

    worker.onerror = (err) => {
      setGpuState('error');
      setGpuStatus(`Worker error: ${err.message}`);
      worker.terminate();
    };

    worker.postMessage({ type: 'run' });
  }, []);

  const runBoth = useCallback(() => {
    runCpu();
    if (hasWebGPU) runGpu();
  }, [runCpu, runGpu, hasWebGPU]);

  return (
    <div style={{ fontFamily: 'monospace', maxWidth: 900, margin: '0 auto', padding: 20 }}>
      <h1 style={{ fontSize: 24 }}>Plonky2 WASM Benchmark</h1>
      <p style={{ color: '#666', marginBottom: 8 }}>
        CPU (PoseidonGoldilocksConfig) vs WebGPU (PoseidonGoldilocksWebGpuConfig)
      </p>
      <p style={{ color: '#666', fontSize: 12 }}>
        Circuit: DummyPsyTypeCCircuit + DummyPsyTypeCRecursiveVerifierCircuit (psy_bench_recursion)
      </p>

      <div style={{ marginBottom: 16 }}>
        <span style={{
          display: 'inline-block',
          padding: '2px 8px',
          borderRadius: 4,
          fontSize: 12,
          background: hasWebGPU === null ? '#888' : hasWebGPU ? '#2a7' : '#c44',
          color: '#fff',
          marginRight: 8,
        }}>
          WebGPU: {hasWebGPU === null ? 'checking...' : hasWebGPU ? 'supported' : 'not available'}
        </span>
      </div>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        <button
          onClick={runCpu}
          disabled={cpuState === 'running'}
          style={btnStyle}
        >
          {cpuState === 'running' ? 'Running CPU...' : 'Run CPU Only'}
        </button>
        <button
          onClick={runGpu}
          disabled={gpuState === 'running' || !hasWebGPU}
          style={btnStyle}
        >
          {gpuState === 'running' ? 'Running WebGPU...' : 'Run WebGPU Only'}
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
          {PHASES.map(({ key, label }) => (
            <tr key={key} style={{ borderBottom: '1px solid #ddd' }}>
              <td style={tdStyle}>{label}</td>
              <td style={{ ...tdStyle, textAlign: 'right' }}>
                {cpuResult ? formatMs((cpuResult as any)[key]) : '-'}
              </td>
              <td style={{ ...tdStyle, textAlign: 'right' }}>
                {gpuResult ? formatMs((gpuResult as any)[key]) : '-'}
              </td>
              <td style={{ ...tdStyle, textAlign: 'right', fontWeight: key === 'total_ms' ? 'bold' : 'normal' }}>
                {cpuResult && gpuResult
                  ? speedup((cpuResult as any)[key], (gpuResult as any)[key])
                  : '-'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {/* Errors */}
      {cpuResult?.error && (
        <div style={{ color: '#c44', marginTop: 12, whiteSpace: 'pre-wrap', fontSize: 12 }}>
          <strong>CPU Error:</strong> {cpuResult.error}
        </div>
      )}
      {gpuResult?.error && (
        <div style={{ color: '#c44', marginTop: 12, whiteSpace: 'pre-wrap', fontSize: 12 }}>
          <strong>WebGPU Error:</strong> {gpuResult.error}
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
          <li>WebGPU requires Chrome 113+ or Edge 113+ with WebGPU enabled</li>
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
