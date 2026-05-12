export function reportError(
  error: unknown,
  modelPath: string,
  mmprojPath?: string
): void {
  console.error("\nExample failed.");
  console.error(`Model path: ${modelPath}`);
  if (mmprojPath) {
    console.error(`mmproj path: ${mmprojPath}`);
  }
  console.error(formatError(error));
}

function formatError(error: unknown): string {
  if (!(error instanceof Error)) {
    return String(error);
  }

  const lines = [`${error.name}: ${error.message}`];
  let cause = error.cause;

  while (cause) {
    if (cause instanceof Error) {
      lines.push(`Caused by: ${cause.name}: ${cause.message}`);
      cause = cause.cause;
    } else {
      lines.push(`Caused by: ${String(cause)}`);
      break;
    }
  }

  return lines.join("\n");
}
