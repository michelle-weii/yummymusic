# Modal API Setup Required

## Current Status

Your client is now ready to send user inputs (text prompt, reference image, and audio file) to your Modal API. However, **your Modal API needs to be updated** to accept and process these user inputs.

## What Needs to Change in Your Modal Code

Currently, your Modal API (`wans2v.py`) uses hardcoded values:

```python
# Hard-coded inputs (CURRENT)
PROMPT = "a white cat wearing sunglasses on a surfboard"
REF_IMAGE = "/wan22/examples/input_image.jpg"
AUDIO = "/wan22/examples/input_audio.MP3"
```

### Required Changes

1. **Update the FastAPI endpoint** to accept form data inputs
2. **Save uploaded files** to temporary paths
3. **Use dynamic inputs** instead of hardcoded values

### Example Modal API Update

```python
@modal.fastapi_endpoint(method="POST")
def api_generate(
    prompt: str = Form(...),
    referenceImage: UploadFile = File(...),
    audioFile: UploadFile = File(...)
):
    import tempfile
    import os
    
    # Save uploaded files to temporary paths
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{referenceImage.filename.split('.')[-1]}") as ref_tmp:
        ref_tmp.write(await referenceImage.read())
        ref_image_path = ref_tmp.name
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audioFile.filename.split('.')[-1]}") as audio_tmp:
        audio_tmp.write(await audioFile.read())
        audio_path = audio_tmp.name
    
    # Use dynamic inputs instead of hardcoded ones
    out_path = _generate_impl(
        prompt=prompt,
        ref_image=ref_image_path, 
        audio=audio_path
    )
    
    # Clean up temp files
    os.unlink(ref_image_path)
    os.unlink(audio_path)
    
    # Return video
    try:
        from fastapi.responses import FileResponse
        return FileResponse(path=out_path, media_type="video/mp4", filename="out.mp4")
    except Exception:
        data = Path(out_path).read_bytes()
        from fastapi import Response
        return Response(content=data, media_type="video/mp4")
```

### Required Imports

Add these imports to your Modal file:

```python
from fastapi import Form, File, UploadFile
```

### Update _generate_impl Function

Modify your `_generate_impl()` function to accept parameters:

```python
def _generate_impl(prompt: str, ref_image: str, audio: str):
    # Use the provided parameters instead of global constants
    # Update your workflow JSON injection logic here
    # Example: prompt_obj["6"]["inputs"]["text"] = prompt
```

## Testing

Once you've updated your Modal API:

1. Deploy the updated Modal function
2. Test the client at `http://localhost:3000`
3. Upload your own images and audio files
4. Watch personalized videos get generated!

## Current Behavior

Until you update the Modal API, the client will send the user inputs but the Modal API will ignore them and continue using hardcoded values.
