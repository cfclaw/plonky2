import React, { useState, useEffect, useCallback } from 'react';

interface BenchmarkResult {
  prove_time_ms: number;
  verify_time_ms: number;
  proof_size: number;
  public_inputs: string[];
  error?: string;
}

interface WasmModule {
  prove_fibonacci: (steps: number) => string;
  health_check: () => string;
}

interface WebGpuWasmModule {
  init_webgpu: () => Promise<void>;
  prove_fibonacci_webgpu: (steps: number) => string;
  health_check: () => string;
}

const styles = {
  container: {
    maxWidth: '900px',
    margin: '0 auto',
    padding: '24px',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace',
    color: '#e0e0e0',
    backgroundColor: '#1a1a2e',
    minHeight: '100vh',
  } as React.CSSProperties,
  header: {
    textAlign: 'center' as const,
    marginBottom: '32px',
  },
  title: {
    fontSize: '28px',
    fontWeight: 700,
    color: '#7c3aed',
    marginBottom: '8px',
  },
  subtitle: {
    fontSize: '14px',
    color: '#888',
  },
  controls: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '16px',
    marginBottom: '32px',
    flexWrap: 'wrap' as const,
  },
  label: {
    fontSize: '14px',
    color: '#aaa',
  },
  input: {
    width: '100px',
    padding: '8px 12px',
    borderRadius: '6px',
    border: '1px solid #444',
    backgroundColor: '#2d2d44',
    color: '#e0e0e0',
    fontSize: '14px',
  } as React.CSSProperties,
  button: {
    padding: '10px 20px',
    borderRadius: '6px',
    border: 'none',
    fontSize: '14px',
    fontWeight: 600,
    cursor: 'pointer',
    transition: 'opacity 0.2s',
  } as React.CSSProperties,
  cpuButton: {
    backgroundColor: '#2563eb',
    color: 'white',
  },
  gpuButton: {
    backgroundColor: '#7c3aed',
    color: 'white',
  },
  bothButton: {
    backgroundColor: '#059669',
    color: 'white',
  },
  disabledButton: {
    opacity: 0.5,
    cursor: 'not-allowed',
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '24px',
    marginTop: '24px',
  } as React.CSSProperties,
  card: {
    backgroundColor: '#16213e',
    borderRadius: '12px',
    padding: '20px',
    border: '1px solid #2d2d44',
  },
  cardTitle: {
    fontSize: '18px',
    fontWeight: 600,
    marginBottom: '16px',
  },
  resultRow: {
    display: 'flex',
    justifyContent: 'space-between',
    padding: '8px 0',
    borderBottom: '1px solid #2d2d44',
    fontSize: '14px',
  } as React.CSSProperties,
  resultLabel: {
    color: '#888',
  },
  resultValue: {
    fontWeight: 600,
    fontFamily: 'monospace',
  },
  status: {
    textAlign: 'center' as const,
    padding: '12px',
    borderRadius: '8px',
    marginBottom: '16px',
    fontSize: '14px',
  },
  loading: {
    backgroundColor: '#1e3a5f',
    color: '#60a5fa',
  },
  success: {
    backgroundColor: '#1a3d2a',
    color: '#4ade80',
  },
  errorStyle: {
    backgroundColor: '#3d1a1a',
    color: '#f87171',
  },
  comparison: {
    backgroundColor: '#16213e',
    borderRadius: '12px',
    padding: '20px',
    border: '1px solid #7c3aed',
    marginTop: '24px',
  },
  speedup: {
    fontSize: '24px',
    fontWeight: 700,
    textAlign: 'center' as const,
    padding: '16px',
  },
};

function App() {
  const [cpuModule, setCpuModule] = useState<WasmModule | null>(null);
  const [gpuModule, setGpuModule] = useState<WebGpuWasmModule | null>(null);
  const [gpuReady, setGpuReady] = useState(false);
  const [webgpuSupported, setWebgpuSupported] = useState<boolean | null>(null);
  const [numSteps, setNumSteps] = useState(100);
  const [cpuResult, setCpuResult] = useState<BenchmarkResult | null>(null);
  const [gpuResult, setGpuResult] = useState<BenchmarkResult | null>(null);
  const [running, setRunning] = useState<string | null>(null);
  const [status, setStatus] = useState<string>('Loading WASM modules...');
  const [statusType, setStatusType] = useState<'loading' | 'success' | 'error'>('loading');

  // Check WebGPU support
  useEffect(() => {
    const hasWebGPU = 'gpu' in navigator;
    setWebgpuSupported(hasWebGPU);
  }, []);

  // Load CPU WASM module
  useEffect(() => {
    (async () => {
      try {
        const mod = await import('./wasm/cpu-prover/plonky2_wasm_prover.js');
        await mod.default();
        setCpuModule(mod as unknown as WasmModule);
        setStatus('CPU WASM module loaded');
        setStatusType('success');
      } catch (e) {
        console.error('Failed to load CPU WASM:', e);
        setStatus(`CPU WASM load failed: ${e}`);
        setStatusType('error');
      }
    })();
  }, []);

  // Load WebGPU WASM module
  useEffect(() => {
    if (!webgpuSupported) return;
    (async () => {
      try {
        const mod = await import('./wasm/webgpu-prover/plonky2_webgpu_prover.js');
        await mod.default();
        setGpuModule(mod as unknown as WebGpuWasmModule);
        setStatus('Initializing WebGPU...');
        setStatusType('loading');
        await mod.init_webgpu();
        setGpuReady(true);
        setStatus('Both modules ready');
        setStatusType('success');
      } catch (e) {
        console.error('Failed to load WebGPU WASM:', e);
        setStatus(`WebGPU WASM load failed: ${e}`);
        setStatusType('error');
      }
    })();
  }, [webgpuSupported]);

  const runCpuBenchmark = useCallback(async () => {
    if (!cpuModule) return;
    setRunning('cpu');
    setCpuResult(null);
    setStatus('Running CPU proof...');
    setStatusType('loading');

    // Yield to UI before heavy computation
    await new Promise(r => setTimeout(r, 50));

    try {
      const resultJson = cpuModule.prove_fibonacci(numSteps);
      const result: BenchmarkResult = JSON.parse(resultJson);
      setCpuResult(result);
      setStatus('CPU proof complete');
      setStatusType('success');
    } catch (e: any) {
      setCpuResult({ prove_time_ms: 0, verify_time_ms: 0, proof_size: 0, public_inputs: [], error: e.message || String(e) });
      setStatus(`CPU proof failed: ${e.message || e}`);
      setStatusType('error');
    }
    setRunning(null);
  }, [cpuModule, numSteps]);

  const runGpuBenchmark = useCallback(async () => {
    if (!gpuModule || !gpuReady) return;
    setRunning('gpu');
    setGpuResult(null);
    setStatus('Running WebGPU proof...');
    setStatusType('loading');

    await new Promise(r => setTimeout(r, 50));

    try {
      const resultJson = gpuModule.prove_fibonacci_webgpu(numSteps);
      const result: BenchmarkResult = JSON.parse(resultJson);
      setGpuResult(result);
      setStatus('WebGPU proof complete');
      setStatusType('success');
    } catch (e: any) {
      setGpuResult({ prove_time_ms: 0, verify_time_ms: 0, proof_size: 0, public_inputs: [], error: e.message || String(e) });
      setStatus(`WebGPU proof failed: ${e.message || e}`);
      setStatusType('error');
    }
    setRunning(null);
  }, [gpuModule, gpuReady, numSteps]);

  const runBothBenchmarks = useCallback(async () => {
    await runCpuBenchmark();
    if (gpuReady) {
      await runGpuBenchmark();
    }
  }, [runCpuBenchmark, runGpuBenchmark, gpuReady]);

  const speedup = cpuResult && gpuResult && !cpuResult.error && !gpuResult.error && gpuResult.prove_time_ms > 0
    ? (cpuResult.prove_time_ms / gpuResult.prove_time_ms).toFixed(2)
    : null;

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <div style={styles.title}>Plonky2 Browser Proving Benchmark</div>
        <div style={styles.subtitle}>
          CPU-only WASM vs WebGPU-accelerated WASM | Fibonacci Circuit | PoseidonGoldilocksConfig
        </div>
      </div>

      <div style={{
        ...styles.status,
        ...(statusType === 'loading' ? styles.loading : statusType === 'success' ? styles.success : styles.errorStyle),
      }}>
        {status}
      </div>

      <div style={styles.controls}>
        <span style={styles.label}>Fibonacci steps:</span>
        <input
          type="number"
          value={numSteps}
          onChange={(e) => setNumSteps(Math.max(1, parseInt(e.target.value) || 1))}
          style={styles.input}
          min={1}
          max={10000}
        />
        <button
          onClick={runCpuBenchmark}
          disabled={!cpuModule || running !== null}
          style={{
            ...styles.button,
            ...styles.cpuButton,
            ...(!cpuModule || running !== null ? styles.disabledButton : {}),
          }}
        >
          {running === 'cpu' ? 'Running...' : 'Run CPU'}
        </button>
        <button
          onClick={runGpuBenchmark}
          disabled={!gpuReady || running !== null}
          style={{
            ...styles.button,
            ...styles.gpuButton,
            ...(!gpuReady || running !== null ? styles.disabledButton : {}),
          }}
        >
          {running === 'gpu' ? 'Running...' : webgpuSupported === false ? 'No WebGPU' : 'Run WebGPU'}
        </button>
        <button
          onClick={runBothBenchmarks}
          disabled={!cpuModule || running !== null}
          style={{
            ...styles.button,
            ...styles.bothButton,
            ...(!cpuModule || running !== null ? styles.disabledButton : {}),
          }}
        >
          Run Both
        </button>
      </div>

      <div style={styles.grid}>
        <ResultCard
          title="CPU-only WASM"
          color="#2563eb"
          result={cpuResult}
          running={running === 'cpu'}
        />
        <ResultCard
          title="WebGPU WASM"
          color="#7c3aed"
          result={gpuResult}
          running={running === 'gpu'}
          unavailable={webgpuSupported === false}
        />
      </div>

      {speedup && (
        <div style={styles.comparison}>
          <div style={{ ...styles.cardTitle, textAlign: 'center' }}>
            Comparison
          </div>
          <div style={{
            ...styles.speedup,
            color: parseFloat(speedup) > 1 ? '#4ade80' : '#f87171',
          }}>
            WebGPU is {speedup}x {parseFloat(speedup) > 1 ? 'faster' : 'slower'} than CPU
          </div>
          <div style={styles.resultRow}>
            <span style={styles.resultLabel}>CPU Prove Time</span>
            <span style={styles.resultValue}>{cpuResult!.prove_time_ms.toFixed(1)} ms</span>
          </div>
          <div style={styles.resultRow}>
            <span style={styles.resultLabel}>WebGPU Prove Time</span>
            <span style={styles.resultValue}>{gpuResult!.prove_time_ms.toFixed(1)} ms</span>
          </div>
          <div style={styles.resultRow}>
            <span style={styles.resultLabel}>Time Saved</span>
            <span style={styles.resultValue}>
              {(cpuResult!.prove_time_ms - gpuResult!.prove_time_ms).toFixed(1)} ms
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

function ResultCard({
  title,
  color,
  result,
  running,
  unavailable,
}: {
  title: string;
  color: string;
  result: BenchmarkResult | null;
  running: boolean;
  unavailable?: boolean;
}) {
  return (
    <div style={styles.card}>
      <div style={{ ...styles.cardTitle, color }}>{title}</div>
      {unavailable ? (
        <div style={{ color: '#666', textAlign: 'center', padding: '20px' }}>
          WebGPU not available in this browser.
          <br />Use Chrome 113+ or Edge 113+.
        </div>
      ) : running ? (
        <div style={{ color: '#60a5fa', textAlign: 'center', padding: '20px' }}>
          Generating proof...
        </div>
      ) : result ? (
        result.error ? (
          <div style={{ color: '#f87171', padding: '8px' }}>Error: {result.error}</div>
        ) : (
          <>
            <div style={styles.resultRow}>
              <span style={styles.resultLabel}>Prove Time</span>
              <span style={{ ...styles.resultValue, color }}>
                {result.prove_time_ms.toFixed(1)} ms
              </span>
            </div>
            <div style={styles.resultRow}>
              <span style={styles.resultLabel}>Verify Time</span>
              <span style={styles.resultValue}>{result.verify_time_ms.toFixed(1)} ms</span>
            </div>
            <div style={styles.resultRow}>
              <span style={styles.resultLabel}>Proof Size</span>
              <span style={styles.resultValue}>
                {(result.proof_size / 1024).toFixed(1)} KB
              </span>
            </div>
            <div style={styles.resultRow}>
              <span style={styles.resultLabel}>Public Inputs</span>
              <span style={styles.resultValue}>{result.public_inputs.length}</span>
            </div>
          </>
        )
      ) : (
        <div style={{ color: '#666', textAlign: 'center', padding: '20px' }}>
          Click a button to run benchmark
        </div>
      )}
    </div>
  );
}

export default App;
