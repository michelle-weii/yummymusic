# YummyMusic Client

Simple React + Vite app to submit a music audio file and a text prompt to a Modal-hosted API that drives ComfyUI (WAN 2.2 s2v). Shows queue state and then renders the resulting video.

## Setup

1. Create `.env` in this folder:

```
VITE_API_BASE_URL=https://your-modal-app.modal.run
```

2. Install and run:

```
npm install
npm run dev
```

## API Contract

- POST `${VITE_API_BASE_URL}/api/wan22/submit`
  - multipart/form-data fields: `audio` (file), `prompt` (string)
  - response: `{ jobId: string }`

- GET `${VITE_API_BASE_URL}/api/wan22/status/:jobId`
  - response: one of
    - `{ state: "queued", position: number }`
    - `{ state: "processing" }`
    - `{ state: "completed", videoUrl: string }`
    - `{ state: "failed", error: string }`

Adjust endpoints if your backend differs.
