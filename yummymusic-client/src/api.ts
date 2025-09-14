import { APP_CONFIG, assertApiBaseUrl } from "./config";

export type SubmitJobResponse = {
  jobId: string;
};

export type QueueStatus =
  | { state: "queued"; position: number }
  | { state: "processing" }
  | { state: "completed"; videoUrl: string }
  | { state: "failed"; error: string };

export async function submitJob(
  audioFile: File,
  prompt: string
): Promise<SubmitJobResponse> {
  const base = assertApiBaseUrl();
  const formData = new FormData();
  formData.append("audio", audioFile);
  formData.append("prompt", prompt);

  const resp = await fetch(`${base}/api/wan22/submit`, {
    method: "POST",
    body: formData,
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Submit failed: ${resp.status} ${text}`);
  }
  return (await resp.json()) as SubmitJobResponse;
}

export async function getQueueStatus(jobId: string): Promise<QueueStatus> {
  const base = assertApiBaseUrl();
  const resp = await fetch(`${base}/api/wan22/status/${encodeURIComponent(jobId)}`);
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Status failed: ${resp.status} ${text}`);
  }
  return (await resp.json()) as QueueStatus;
}


