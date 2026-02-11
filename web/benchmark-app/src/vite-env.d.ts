/// <reference types="vite/client" />

declare module '*.js' {
  const value: any;
  export default value;
  export const prove_fibonacci: (steps: number) => string;
  export const prove_fibonacci_webgpu: (steps: number) => string;
  export const init_webgpu: () => Promise<void>;
  export const health_check: () => string;
}
