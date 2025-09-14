import { useEffect, useRef, useState } from "react";
import { APP_CONFIG } from "../config";
import { QueueStatus as QueueState, getQueueStatus } from "../api";

type Props = {
  jobId: string;
  onCompleted: (videoUrl: string) => void;
  onFailed?: (error: string) => void;
};

export function QueueStatus({ jobId, onCompleted, onFailed }: Props) {
  const [status, setStatus] = useState<QueueState | null>(null);
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    let isCancelled = false;

    async function poll() {
      try {
        const s = await getQueueStatus(jobId);
        if (isCancelled) return;
        setStatus(s);
        if (s.state === "completed") {
          onCompleted(s.videoUrl);
          return; // stop polling
        }
        if (s.state === "failed") {
          onFailed?.(s.error);
          return;
        }
        timerRef.current = window.setTimeout(poll, APP_CONFIG.queuePollIntervalMs);
      } catch (err) {
        // Backoff on transient errors
        timerRef.current = window.setTimeout(poll, APP_CONFIG.queuePollIntervalMs * 2);
      }
    }

    poll();
    return () => {
      isCancelled = true;
      if (timerRef.current) window.clearTimeout(timerRef.current);
    };
  }, [jobId, onCompleted, onFailed]);

  if (!status) return <div>Checking queue...</div>;
  if (status.state === "queued") return <div>In queue. Position: {status.position}</div>;
  if (status.state === "processing") return <div>Processing...</div>;
  if (status.state === "failed") return <div>Failed: {status.error}</div>;
  return null;
}


