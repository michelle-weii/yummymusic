import { useCallback, useState } from "react";
import "./App.css";
import { UploadForm } from "./components/UploadForm";
import { QueueStatus } from "./components/QueueStatus";
import { VideoPlayer } from "./components/VideoPlayer";
import { submitJob } from "./api";

function App() {
  const [jobId, setJobId] = useState<string | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = useCallback(
    async ({ audioFile, prompt }: { audioFile: File; prompt: string }) => {
      setError(null);
      setVideoUrl(null);
      setJobId(null);
      const { jobId } = await submitJob(audioFile, prompt);
      setJobId(jobId);
    },
    []
  );

  return (
    <div style={{ maxWidth: 960, margin: "0 auto", padding: 24 }}>
      <h1>YummyMusic Video Generator</h1>
      <UploadForm onSubmit={handleSubmit} />
      {jobId && !videoUrl && (
        <div style={{ marginTop: 24 }}>
          <h2>Queue</h2>
          <QueueStatus
            jobId={jobId}
            onCompleted={(url) => setVideoUrl(url)}
            onFailed={(e) => setError(e)}
          />
        </div>
      )}
      {error && (
        <div style={{ color: "#b00020", marginTop: 12 }}>Error: {error}</div>
      )}
      {videoUrl && (
        <div style={{ marginTop: 24 }}>
          <h2>Result</h2>
          <VideoPlayer src={videoUrl} />
        </div>
      )}
    </div>
  );
}

export default App;
