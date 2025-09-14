'use client'

import { useState, useRef } from 'react'

const API_URL = '/api/generate'

type GenerationState = 'idle' | 'generating' | 'completed' | 'error'

interface FormData {
  prompt: string
  referenceImage: File | null
  audioFile: File | null
}

export default function Home() {
  const [state, setState] = useState<GenerationState>('idle')
  const [videoUrl, setVideoUrl] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [formData, setFormData] = useState<FormData>({
    prompt: 'a white cat wearing sunglasses on a surfboard',
    referenceImage: null,
    audioFile: null
  })

  const imageInputRef = useRef<HTMLInputElement>(null)
  const audioInputRef = useRef<HTMLInputElement>(null)

  const handleInputChange = (field: keyof FormData, value: string | File | null) => {
    setFormData(prev => ({ ...prev, [field]: value }))
  }

  const generateVideo = async () => {
    // Validate inputs
    if (!formData.prompt.trim()) {
      setError('Please enter a text prompt')
      return
    }
    if (!formData.referenceImage) {
      setError('Please upload a reference image')
      return
    }
    if (!formData.audioFile) {
      setError('Please upload an audio file')
      return
    }

    setState('generating')
    setError(null)
    setVideoUrl(null)

    try {
      // Create FormData for multipart/form-data upload
      const uploadData = new FormData()
      uploadData.append('prompt', formData.prompt)
      uploadData.append('referenceImage', formData.referenceImage)
      uploadData.append('audioFile', formData.audioFile)

      const response = await fetch(API_URL, {
        method: 'POST',
        body: uploadData,
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      // The API returns a video file, so we create a blob URL
      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      setVideoUrl(url)
      setState('completed')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
      setState('error')
    }
  }

  const resetGeneration = () => {
    setState('idle')
    setVideoUrl(null)
    setError(null)
    // Reset form
    setFormData({
      prompt: 'a white cat wearing sunglasses on a surfboard',
      referenceImage: null,
      audioFile: null
    })
    // Reset file inputs
    if (imageInputRef.current) imageInputRef.current.value = ''
    if (audioInputRef.current) audioInputRef.current.value = ''
  }

  return (
    <div className="app">
      <div className="container">
        <h1>🎵 YummyMusic Video Generator</h1>
        <p className="subtitle">Generate amazing videos with AI</p>

        {state === 'idle' && (
          <div className="input-form">
            {/* Text Prompt Input */}
            <div className="input-group">
              <label htmlFor="prompt" className="input-label">
                🎯 Text Prompt
              </label>
              <textarea
                id="prompt"
                className="text-input"
                value={formData.prompt}
                onChange={(e) => handleInputChange('prompt', e.target.value)}
                placeholder="Describe what you want to see in your video..."
                rows={3}
              />
            </div>

            {/* Reference Image Input */}
            <div className="input-group">
              <label htmlFor="referenceImage" className="input-label">
                🖼️ Reference Image
              </label>
              <div className="file-input-wrapper">
                <input
                  ref={imageInputRef}
                  id="referenceImage"
                  type="file"
                  className="file-input"
                  accept="image/*"
                  onChange={(e) => {
                    const file = e.target.files?.[0] || null
                    handleInputChange('referenceImage', file)
                  }}
                />
                <div className="file-input-display">
                  {formData.referenceImage ? (
                    <div className="file-selected">
                      ✅ {formData.referenceImage.name}
                    </div>
                  ) : (
                    <div className="file-placeholder">
                      Choose an image file (JPG, PNG, etc.)
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Audio File Input */}
            <div className="input-group">
              <label htmlFor="audioFile" className="input-label">
                🎵 Audio File
              </label>
              <div className="file-input-wrapper">
                <input
                  ref={audioInputRef}
                  id="audioFile"
                  type="file"
                  className="file-input"
                  accept="audio/*"
                  onChange={(e) => {
                    const file = e.target.files?.[0] || null
                    handleInputChange('audioFile', file)
                  }}
                />
                <div className="file-input-display">
                  {formData.audioFile ? (
                    <div className="file-selected">
                      ✅ {formData.audioFile.name}
                    </div>
                  ) : (
                    <div className="file-placeholder">
                      Choose an audio file (MP3, WAV, etc.)
                    </div>
                  )}
                </div>
              </div>
            </div>

            <button className="generate-btn" onClick={generateVideo}>
              Generate Video
            </button>
          </div>
        )}

        {state === 'generating' && (
          <div className="loading-container">
            <div className="spinner"></div>
            <p className="loading-text">Generating your video...</p>
            <p className="loading-subtext">This process typically takes 10-30 minutes. Please be patient!</p>
            <div className="progress-info">
              <p>🔄 Processing your inputs with the Wan AI model</p>
              <p>🎬 Creating high-quality video synchronized with your audio</p>
              <p>⏰ Expected completion: 15-25 minutes</p>
            </div>
            <div className="generation-inputs">
              <p className="input-preview"><strong>Prompt:</strong> {formData.prompt}</p>
              <p className="input-preview"><strong>Image:</strong> {formData.referenceImage?.name}</p>
              <p className="input-preview"><strong>Audio:</strong> {formData.audioFile?.name}</p>
            </div>
          </div>
        )}

        {state === 'error' && (
          <div className="error-container">
            <p className="error-text">❌ Error: {error}</p>
            <button className="retry-btn" onClick={generateVideo}>
              Retry
            </button>
            <button className="reset-btn" onClick={resetGeneration}>
              Reset
            </button>
          </div>
        )}

        {state === 'completed' && videoUrl && (
          <div className="success-container">
            <p className="success-text">✅ Video generated successfully!</p>
            <video
              className="generated-video"
              src={videoUrl}
              controls
              autoPlay
              loop
            >
              Your browser does not support the video tag.
            </video>
            <div className="generation-summary">
              <p className="input-preview"><strong>Prompt:</strong> {formData.prompt}</p>
              <p className="input-preview"><strong>Image:</strong> {formData.referenceImage?.name}</p>
              <p className="input-preview"><strong>Audio:</strong> {formData.audioFile?.name}</p>
            </div>
            <button className="generate-another-btn" onClick={resetGeneration}>
              Generate Another
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
