# YummyMusic Client

A Next.js client for generating videos using the YummyMusic Modal API with CORS-free server-side proxy.

## Features

- Simple one-button interface
- Real-time loading states
- Video playback when generation is complete
- Error handling and retry functionality
- Responsive design
- **CORS-free API calls** via server-side proxy

## Architecture

- **Frontend**: Next.js App Router with React client components
- **Backend**: Next.js API route that proxies requests to Modal
- **Flow**: Browser → Local API (`/api/generate`) → Modal API → Browser

This eliminates CORS issues by having the server make the Modal API call instead of the browser.

## Getting Started

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start the development server:
   ```bash
   npm run dev
   ```

3. Open http://localhost:3000 in your browser

## Usage

1. **Enter a text prompt** describing what you want to see in your video
2. **Upload a reference image** (JPG, PNG, etc.) to guide the visual style
3. **Upload an audio file** (MP3, WAV, etc.) to sync the video with
4. **Click "Generate Video"** and wait for the AI to create your video
5. **Watch your generated video** in the built-in player
6. **Generate another** with different inputs or start over

## Input Requirements

- **Text Prompt**: Describe your desired video content (required)
- **Reference Image**: Visual reference for the AI (required)
- **Audio File**: Audio track to synchronize with (required)

## API Architecture

- **Client**: Calls `/api/generate` (local Next.js API route)
- **Server**: Proxies requests to Modal API endpoint
- **Modal API**: `https://regex-golf--wan2-2-s2v-14b-yummymusic-api-generate-dev.modal.run`

## Files Structure

```
app/
├── api/generate/route.ts    # Server-side proxy to Modal API
├── layout.tsx              # Root layout
├── page.tsx               # Main client component
└── globals.css           # Global styles
```

## Build

To build for production:
```bash
npm run build
npm start
```
