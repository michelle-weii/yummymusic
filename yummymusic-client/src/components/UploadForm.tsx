import { useRef, useState } from "react";

type Props = {
  onSubmit: (data: { audioFile: File; prompt: string }) => Promise<void> | void;
};

export function UploadForm({ onSubmit }: Props) {
  const [prompt, setPrompt] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const audioInputRef = useRef<HTMLInputElement | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!audioInputRef.current || !audioInputRef.current.files || audioInputRef.current.files.length === 0) {
      alert("Please select an audio file.");
      return;
    }
    try {
      setSubmitting(true);
      const file = audioInputRef.current.files[0];
      await onSubmit({ audioFile: file, prompt });
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} style={{ display: "grid", gap: 12 }}>
      <label>
        Music audio file
        <input ref={audioInputRef} type="file" accept="audio/*" required />
      </label>
      <label>
        Text prompt
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Describe the desired video"
          rows={4}
          required
        />
      </label>
      <button type="submit" disabled={submitting}>
        {submitting ? "Submitting..." : "Generate video"}
      </button>
    </form>
  );
}


