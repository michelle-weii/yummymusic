import { NextRequest, NextResponse } from 'next/server'

const MODAL_API_URL = 'https://regex-golf--wan2-2-s2v-14b-yummymusic-fastapi-app-dev.modal.run'

export async function POST(request: NextRequest) {
  console.log('🚀 API route started')
  
  try {
    // Parse the multipart form data
    console.log('📝 Parsing form data...')
    const formData = await request.formData()
    const prompt = formData.get('prompt') as string
    const referenceImage = formData.get('referenceImage') as File
    const audioFile = formData.get('audioFile') as File

    console.log('📋 Form data parsed:', {
      prompt: prompt ? `"${prompt.substring(0, 50)}..."` : 'MISSING',
      referenceImage: referenceImage ? `${referenceImage.name} (${referenceImage.size} bytes)` : 'MISSING',
      audioFile: audioFile ? `${audioFile.name} (${audioFile.size} bytes)` : 'MISSING',
    })

    // Validate inputs
    if (!prompt || !referenceImage || !audioFile) {
      console.error('❌ Validation failed - missing inputs')
      return NextResponse.json(
        { error: 'Missing required inputs: prompt, referenceImage, or audioFile' },
        { status: 400 }
      )
    }

    // Create FormData to forward to Modal API
    console.log('📦 Creating form data for Modal API...')
    const modalFormData = new FormData()
    modalFormData.append('prompt', prompt)
    modalFormData.append('referenceImage', referenceImage)
    modalFormData.append('audioFile', audioFile)

    console.log(`🌐 Calling Modal API at: ${MODAL_API_URL}`)
    
    // Forward the request to Modal API with extended timeout for video generation
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 1800000) // 30 minute timeout for video generation
    
    try {
      const response = await fetch(MODAL_API_URL, {
        method: 'POST',
        body: modalFormData,
        signal: controller.signal,
        // Add headers that might be needed
        headers: {
          // Don't set Content-Type for FormData, let fetch set it
        }
      })
      
      clearTimeout(timeoutId)
      
      console.log(`📡 Modal API response status: ${response.status}`)
      console.log(`📡 Modal API response headers:`, Object.fromEntries(response.headers.entries()))

      if (!response.ok) {
        const errorText = await response.text()
        console.error('❌ Modal API error:', {
          status: response.status,
          statusText: response.statusText,
          errorText: errorText.substring(0, 1000) + (errorText.length > 1000 ? '...' : '')
        })
        
        return NextResponse.json(
          { 
            error: `Modal API error: ${response.status} ${response.statusText}`,
            details: errorText.substring(0, 500)
          },
          { status: response.status }
        )
      }

      // Get the video blob from Modal
      console.log('🎥 Processing video response...')
      const videoBlob = await response.blob()
      console.log(`✅ Video blob received: ${videoBlob.size} bytes, type: ${videoBlob.type}`)

      // Return the video with proper headers
      return new NextResponse(videoBlob, {
        status: 200,
        headers: {
          'Content-Type': 'video/mp4',
          'Content-Disposition': 'inline; filename="generated-video.mp4"',
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'POST',
          'Access-Control-Allow-Headers': 'Content-Type',
        },
      })
      
    } catch (fetchError: any) {
      clearTimeout(timeoutId)
      
      if (fetchError.name === 'AbortError') {
        console.error('⏰ Modal API request timed out after 30 minutes')
        return NextResponse.json(
          { error: 'Video generation timed out after 30 minutes. The process may still be running on the server.' },
          { status: 408 }
        )
      }
      
      console.error('🌐 Network error calling Modal API:', fetchError)
      return NextResponse.json(
        { 
          error: 'Network error calling Modal API',
          details: fetchError.message
        },
        { status: 503 }
      )
    }
    
  } catch (error: any) {
    console.error('💥 Unexpected error in API route:', error)
    console.error('Stack trace:', error.stack)
    
    return NextResponse.json(
      { 
        error: 'Internal server error',
        details: error.message,
        type: error.constructor.name
      },
      { status: 500 }
    )
  }
}

// Handle preflight requests
export async function OPTIONS(request: NextRequest) {
  return new NextResponse(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    },
  })
}
