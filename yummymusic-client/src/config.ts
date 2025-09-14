export const APP_CONFIG = {
  apiBaseUrl:
    (import.meta as unknown as { env?: Record<string, string> }).env?.
      VITE_API_BASE_URL ||
    (typeof process !== "undefined"
      ? (process.env?.VITE_API_BASE_URL as string | undefined)
      : undefined) ||
    "",
  // Polling interval in milliseconds for queue status
  queuePollIntervalMs: 2500,
};

export function assertApiBaseUrl(): string {
  if (!APP_CONFIG.apiBaseUrl) {
    throw new Error(
      "VITE_API_BASE_URL is not set. Create .env and set VITE_API_BASE_URL to your Modal/ComfyUI endpoint."
    );
  }
  return APP_CONFIG.apiBaseUrl;
}


