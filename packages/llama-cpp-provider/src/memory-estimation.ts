import { freemem, totalmem } from "node:os";
import type {
  LlamaCppMemorySafetyConfig,
  LlamaCppModelMemoryInfo,
} from "./llama-cpp-provider-config.js";

const GIB = 1024 ** 3;
const DEFAULT_CONTEXT_SIZE = 2048;
const DEFAULT_BYTES_PER_VALUE = 2;

export interface EstimateMemoryUsageOptions {
  model: LlamaCppModelMemoryInfo;
  contextSize: number;
  /**
   * GGUF model file size, if known.
   */
  modelFileSizeBytes?: number;
  /**
   * Multimodal projector file size, if known.
   */
  mmprojFileSizeBytes?: number;
  /**
   * Override bytes per K/V cache element. Defaults to model metadata or f16.
   */
  kvCacheBytesPerValue?: number;
  /**
   * Extra overhead for llama.cpp compute buffers and Metal allocations.
   * Defaults to max(1 GiB, 15% of KV cache).
   */
  computeOverheadBytes?: number;
}

export interface MemoryUsageEstimate {
  contextSize: number;
  kvCacheBytes: number;
  modelBytes: number;
  mmprojBytes: number;
  computeOverheadBytes: number;
  totalBytes: number;
  bytesPerToken: number;
  maxContextSize?: number;
}

export interface MemorySafetyCheckOptions {
  model?: LlamaCppModelMemoryInfo;
  contextSize?: number;
  modelFileSizeBytes?: number;
  mmprojFileSizeBytes?: number;
  memorySafety?: LlamaCppMemorySafetyConfig;
}

export interface MemorySafetyCheckResult {
  contextSize: number;
  estimate?: MemoryUsageEstimate;
}

export function estimateMemoryUsage(
  options: EstimateMemoryUsageOptions
): MemoryUsageEstimate {
  validateContextSize(options.contextSize);

  const bytesPerValue =
    options.kvCacheBytesPerValue ??
    options.model.kvCache.bytesPerValue ??
    DEFAULT_BYTES_PER_VALUE;

  if (!Number.isFinite(bytesPerValue) || bytesPerValue <= 0) {
    throw new Error("kvCacheBytesPerValue must be a positive number");
  }

  const bytesPerToken = options.model.kvCache.layers.reduce((total, layer) => {
    const keyHeads = layer.keyHeads ?? layer.keyValueHeads;
    const valueHeads = layer.valueHeads ?? layer.keyValueHeads;
    const keyHeadDim = layer.keyHeadDim ?? layer.headDim;
    const valueHeadDim = layer.valueHeadDim ?? layer.headDim;

    if (
      layer.count <= 0 ||
      keyHeads === undefined ||
      valueHeads === undefined ||
      keyHeadDim === undefined ||
      valueHeadDim === undefined
    ) {
      throw new Error("Invalid KV-cache layer memory metadata");
    }

    return (
      total +
      layer.count *
        (keyHeads * keyHeadDim + valueHeads * valueHeadDim) *
        bytesPerValue
    );
  }, 0);

  const kvCacheBytes = bytesPerToken * options.contextSize;
  const computeOverheadBytes =
    options.computeOverheadBytes ??
    Math.max(GIB, Math.ceil(kvCacheBytes * 0.15));
  const modelBytes = options.modelFileSizeBytes ?? 0;
  const mmprojBytes = options.mmprojFileSizeBytes ?? 0;

  return {
    contextSize: options.contextSize,
    kvCacheBytes,
    modelBytes,
    mmprojBytes,
    computeOverheadBytes,
    totalBytes: kvCacheBytes + modelBytes + mmprojBytes + computeOverheadBytes,
    bytesPerToken,
    maxContextSize: options.model.maxContextSize,
  };
}

export function checkMemorySafety(
  options: MemorySafetyCheckOptions
): MemorySafetyCheckResult {
  const requestedContextSize = options.contextSize ?? DEFAULT_CONTEXT_SIZE;
  validateContextSize(requestedContextSize);

  const safety = options.memorySafety;
  if (safety?.mode === "off" || !options.model) {
    return { contextSize: requestedContextSize };
  }

  const estimate = estimateMemoryUsage({
    model: options.model,
    contextSize: requestedContextSize,
    modelFileSizeBytes: options.modelFileSizeBytes,
    mmprojFileSizeBytes: options.mmprojFileSizeBytes,
    kvCacheBytesPerValue: safety?.kvCacheBytesPerValue,
    computeOverheadBytes: safety?.computeOverheadBytes,
  });

  const maxContextSize = options.model.maxContextSize;
  if (maxContextSize !== undefined && requestedContextSize > maxContextSize) {
    if (safety?.mode === "clamp") {
      return estimateSafeContextSize({
        ...options,
        contextSize: maxContextSize,
      });
    }

    throw new Error(
      `Requested contextSize ${requestedContextSize.toLocaleString()} exceeds the model ` +
        `maximum context size of ${maxContextSize.toLocaleString()}.`
    );
  }

  const memoryBudgetBytes = resolveMemoryBudgetBytes(safety);
  if (estimate.totalBytes <= memoryBudgetBytes) {
    return { contextSize: requestedContextSize, estimate };
  }

  if (safety?.mode === "clamp") {
    return estimateSafeContextSize(options);
  }

  const safeContextSize = calculateSafeContextSize({
    model: options.model,
    memoryBudgetBytes,
    modelFileSizeBytes: options.modelFileSizeBytes,
    mmprojFileSizeBytes: options.mmprojFileSizeBytes,
    memorySafety: safety,
  });

  throw new Error(
    `Requested contextSize ${requestedContextSize.toLocaleString()} is estimated to require ` +
      `${formatBytes(estimate.totalBytes)} (${formatBytes(estimate.kvCacheBytes)} KV cache). ` +
      `The current safety budget is ${formatBytes(memoryBudgetBytes)}. ` +
      `Use contextSize <= ${safeContextSize.toLocaleString()}, lower memorySafety settings, ` +
      `or set memorySafety: { mode: "off" } to bypass this check.`
  );
}

function estimateSafeContextSize(
  options: MemorySafetyCheckOptions & { model?: LlamaCppModelMemoryInfo }
): MemorySafetyCheckResult {
  if (!options.model) {
    return { contextSize: options.contextSize ?? DEFAULT_CONTEXT_SIZE };
  }

  const requestedContextSize = options.contextSize ?? DEFAULT_CONTEXT_SIZE;
  const memoryBudgetBytes = resolveMemoryBudgetBytes(options.memorySafety);
  const contextSize = calculateSafeContextSize({
    model: options.model,
    memoryBudgetBytes,
    modelFileSizeBytes: options.modelFileSizeBytes,
    mmprojFileSizeBytes: options.mmprojFileSizeBytes,
    memorySafety: options.memorySafety,
  });
  const clampedContextSize = Math.min(requestedContextSize, contextSize);

  return {
    contextSize: clampedContextSize,
    estimate: estimateMemoryUsage({
      model: options.model,
      contextSize: clampedContextSize,
      modelFileSizeBytes: options.modelFileSizeBytes,
      mmprojFileSizeBytes: options.mmprojFileSizeBytes,
      kvCacheBytesPerValue: options.memorySafety?.kvCacheBytesPerValue,
      computeOverheadBytes: options.memorySafety?.computeOverheadBytes,
    }),
  };
}

function calculateSafeContextSize(options: {
  model: LlamaCppModelMemoryInfo;
  memoryBudgetBytes: number;
  modelFileSizeBytes?: number;
  mmprojFileSizeBytes?: number;
  memorySafety?: LlamaCppMemorySafetyConfig;
}): number {
  const maxContextSize =
    options.model.maxContextSize ?? Number.MAX_SAFE_INTEGER;
  let low = 1;
  let high = maxContextSize;
  let best = 1;

  while (low <= high) {
    const mid = Math.floor((low + high) / 2);
    const estimate = estimateMemoryUsage({
      model: options.model,
      contextSize: mid,
      modelFileSizeBytes: options.modelFileSizeBytes,
      mmprojFileSizeBytes: options.mmprojFileSizeBytes,
      kvCacheBytesPerValue: options.memorySafety?.kvCacheBytesPerValue,
      computeOverheadBytes: options.memorySafety?.computeOverheadBytes,
    });

    if (estimate.totalBytes <= options.memoryBudgetBytes) {
      best = mid;
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }

  const oneTokenEstimate = estimateMemoryUsage({
    model: options.model,
    contextSize: 1,
    modelFileSizeBytes: options.modelFileSizeBytes,
    mmprojFileSizeBytes: options.mmprojFileSizeBytes,
    kvCacheBytesPerValue: options.memorySafety?.kvCacheBytesPerValue,
    computeOverheadBytes: options.memorySafety?.computeOverheadBytes,
  });

  if (oneTokenEstimate.totalBytes > options.memoryBudgetBytes) {
    return 1;
  }

  return best;
}

function resolveMemoryBudgetBytes(safety?: LlamaCppMemorySafetyConfig): number {
  if (safety?.maxMemoryBytes !== undefined) {
    return safety.maxMemoryBytes;
  }

  const availableBytes = freemem();
  const totalBytes = totalmem();
  const reserveMemoryBytes =
    safety?.reserveMemoryBytes ??
    Math.min(16 * GIB, Math.max(4 * GIB, Math.floor(totalBytes * 0.1)));
  const utilization = safety?.memoryUtilization ?? 0.9;

  return Math.max(
    0,
    Math.floor(
      Math.min(
        availableBytes * utilization,
        availableBytes - reserveMemoryBytes
      )
    )
  );
}

function validateContextSize(contextSize: number): void {
  if (!Number.isInteger(contextSize) || contextSize <= 0) {
    throw new Error("contextSize must be a positive integer");
  }
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`;
  }

  const units = ["KiB", "MiB", "GiB", "TiB"];
  let value = bytes / 1024;
  let unitIndex = 0;

  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex++;
  }

  return `${value.toFixed(1)} ${units[unitIndex]}`;
}
