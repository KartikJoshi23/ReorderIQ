/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

/** Cross-origin isolation enables SharedArrayBuffer → multi-threaded TensorFlow.js WebAssembly.
 *  The same headers are set for production in netlify.toml. */
const isolationHeaders = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

export default defineConfig({
  plugins: [react(), tailwindcss()],
  worker: { format: "es" },
  server: { headers: isolationHeaders },
  preview: { headers: isolationHeaders },
  // Pre-bundle worker dependencies at start-up so the dev server never reloads mid-session.
  // (tfjs-backend-wasm must be pre-bundled: its ESM entry imports a CommonJS helper.)
  optimizeDeps: { include: ["@tensorflow/tfjs", "@tensorflow/tfjs-backend-wasm", "comlink", "zod"] },
  build: {
    target: "es2022",
    sourcemap: false,
    chunkSizeWarningLimit: 2000,
    // Long-lived vendor chunks: they change rarely, so browsers keep them cached between deploys.
    rollupOptions: { output: { manualChunks: { react: ["react", "react-dom"], motion: ["framer-motion"], echarts: ["echarts/core", "echarts/charts", "echarts/components", "echarts/renderers"] } } },
  },
  test: {
    environment: "node",
    include: ["tests/unit/**/*.test.ts"],
    testTimeout: 30_000,
  },
});
