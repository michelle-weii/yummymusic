import modal, subprocess, os
from pathlib import Path

app = modal.App("wan2.2-s2v-14b-yummymusic")

# Global server process cached across warm invocations
SERVER_PROC = None  # type: ignore[var-annotated]

# HF_REPO = "Wan-AI/Wan2.2-S2V-14B"
# MODEL_DIR = "/models/Wan2.2-S2V-14B"
# WAN_REPO_DIR = "/wan22"
CACHE_VOL = modal.Volume.from_name("wan22-s2v-cache", create_if_missing=True)
WEIGHTS_VOL = modal.Volume.from_name("weights_vol", create_if_missing=True)
WAN_LOCAL_DIR = "/Users/willi1/yummymusic/Wan_2.2_ComfyUI_Repackaged"
COMFY_LOCAL_DIR = "/Users/willi1/yummymusic/ComfyUI"
COMFY_REMOTE_DIR = "/opt/comfyui"

# Only sync weights/examples into the weights volume if explicitly requested
SYNC_WEIGHTS = os.environ.get("SYNC_WEIGHTS", "0") == "1"

# Hard-coded inputs
PROMPT = "a white cat wearing sunglasses on a surfboard"
REF_IMAGE = "/wan22/examples/input_image.jpg"
AUDIO = "/wan22/examples/input_audio.MP3"
OUT_PATH = "/cache/out.mp4"
WORKFLOW_JSON = "/wan22/workflows/Wan2.2Sound2Vid.json"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "ffmpeg",
        "git",
        # Audio libs commonly required by soundfile/sounddevice/librosa
        "libsndfile1",
        "libportaudio2",
    )
    #.uv_pip_install("torch", "torchvision", "torchaudio", "transformers", "diffusers", "huggingface_hub")
    .run_commands(
        "python -m pip install --upgrade pip",
        # CUDA 12.1 wheels (adjust if Modal docs say otherwise)
        "python -m pip install --index-url https://download.pytorch.org/whl/cu121 "
        "torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1",
        # (Optional, often speeds attention kernels)
        "python -m pip install --index-url https://download.pytorch.org/whl/cu121 xformers==0.0.27.post1",
    )
    .add_local_dir(COMFY_LOCAL_DIR, COMFY_REMOTE_DIR, copy=True)
    .run_commands(
        f"python -m pip install --upgrade pip",
        f"python -m pip install -r {COMFY_REMOTE_DIR}/requirements.txt",
        f"python -m pip install openai",
    )
)

def _ensure_server_running(extra_yaml: str):
    global SERVER_PROC
    if SERVER_PROC is not None and SERVER_PROC.poll() is None:
        return
    print("Starting ComfyUI server (persistent)...")
    SERVER_PROC = subprocess.Popen([
        "python", f"{COMFY_REMOTE_DIR}/main.py",
        "--disable-auto-launch",
        "--listen", "127.0.0.1",
        "--port", "8188",
        "--input-directory", "/wan22/examples",
        "--input-directory", "/cache",
        "--extra-model-paths-config", extra_yaml,
    ])


def _enhance_prompt(user_prompt: str) -> str:
    """
    Enhance the user's prompt using OpenAI Assistant for dynamic optimization
    """
    try:
        import openai
        import os
        
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Create a thread and send the prompt to the assistant
        thread = client.beta.threads.create()
        
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=f"Enhance this prompt for music video generation: {user_prompt}"
        )
        
        # Run the assistant
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id="asst_HQK8oDnZbP37p6YZEdPkZJYf"
        )
        
        # Wait for completion and get response
        import time
        while run.status in ['queued', 'in_progress']:
            time.sleep(1)
            run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            
        if run.status == 'completed':
            messages = client.beta.threads.messages.list(thread_id=thread.id)
            enhanced_prompt = messages.data[0].content[0].text.value.strip()
            return enhanced_prompt
        else:
            print(f"Assistant run failed with status: {run.status}")
            # Fallback to basic enhancement
            return f"{user_prompt}, adhere to style of a music video, music, aesthetic, artistic, cinematic lighting, high quality, detailed"
            
    except Exception as e:
        print(f"Error calling OpenAI assistant: {e}")
        # Fallback to basic enhancement if API fails
        return f"{user_prompt}, adhere to style of a music video, music, aesthetic, artistic, cinematic lighting, high quality, detailed"

def _generate_impl(prompt=None, ref_image_path=None, audio_path=None):
    """Run ComfyUI headless, post a workflow, and copy the output file."""
    
    # Use user inputs or fall back to defaults
    if prompt is None:
        prompt = PROMPT
    if ref_image_path is None:
        ref_image_path = REF_IMAGE
    if audio_path is None:
        audio_path = AUDIO
    
    import json
    import time
    from urllib import request
    
    def _write_extra_model_paths_yaml(yaml_path: str):
        # ComfyUI expects a top-level map, each entry is a dict of model_type -> newline-separated paths
        content = "\n".join([
            "wan22:",
            "  audio_encoders: /wan22/split_files/audio_encoders",
            "  diffusion_models: /wan22/split_files/diffusion_models",
            "  text_encoders: /wan22/split_files/text_encoders",
            "  vae: /wan22/split_files/vae",
        ])
        Path(yaml_path).write_text(content)

    def _wait_for_http(url: str, timeout_seconds: int = 180):
        start = time.time()
        while time.time() - start < timeout_seconds:
            try:
                with request.urlopen(url, timeout=5) as resp:
                    if 200 <= resp.status < 500:
                        return True
            except Exception:
                pass
            time.sleep(1)
        return False

    def _post_prompt(prompt_obj: dict):
        data = json.dumps({"prompt": prompt_obj}).encode("utf-8")
        req = request.Request("http://127.0.0.1:8188/prompt", data=data)
        request.urlopen(req)

    def _latest_file_in_dir(directory: Path):
        files = [p for p in directory.glob("**/*") if p.is_file()]
        return max(files, key=lambda p: p.stat().st_mtime) if files else None

    def _latest_video_file_in_dir(directory: Path):
        video_exts = {".mp4", ".webm", ".mov", ".mkv"}
        files = [p for p in directory.glob("**/*") if p.is_file() and p.suffix.lower() in video_exts]
        return max(files, key=lambda p: p.stat().st_mtime) if files else None

    def _wait_until_stable(path: Path, stable_checks: int = 3, sleep_seconds: float = 1.0, timeout: int = 600) -> bool:
        """Wait until file size stops changing (>0) for a few checks."""
        import time as _t
        start = _t.time()
        last_size = -1
        stable_count = 0
        while _t.time() - start < timeout:
            try:
                size = path.stat().st_size
            except FileNotFoundError:
                size = -1
            if size > 0 and size == last_size:
                stable_count += 1
                if stable_count >= stable_checks:
                    return True
            else:
                stable_count = 0
            last_size = size
            _t.sleep(sleep_seconds)
        return False

    def _collect_new_frames(base_dir: Path, since_ts: float):
        image_exts = {".png", ".jpg", ".jpeg", ".webp"}
        frames = [p for p in base_dir.glob("**/*") if p.is_file() and p.suffix.lower() in image_exts and p.stat().st_mtime >= since_ts]
        frames.sort(key=lambda p: (p.parent.as_posix(), p.stat().st_mtime, p.name))
        return frames

    def _encode_frames_to_video(frames: list[Path], fps: int, out_path: Path) -> bool:
        if not frames:
            return False
        # Build a concat list to ensure correct order and support mixed filenames
        import tempfile, subprocess as sp
        with tempfile.TemporaryDirectory() as td:
            list_file = Path(td) / "inputs.txt"
            # Ensure even dimensions later with scale filter
            list_file.write_text("\n".join([f"file '{f.as_posix()}'" for f in frames]) + "\n")
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-r", str(fps),
                "-i", str(list_file),
                "-vf", "fps=%d,scale=trunc(iw/2)*2:trunc(ih/2)*2" % fps,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(out_path),
            ]
            try:
                sp.run(cmd, check=True, stdout=sp.PIPE, stderr=sp.PIPE)
                return out_path.exists() and out_path.stat().st_size > 0
            except Exception:
                return False

    extra_yaml = "/tmp/extra_model_paths.yaml"
    _write_extra_model_paths_yaml(extra_yaml)
    _ensure_server_running(extra_yaml)

    try:
        if not _wait_for_http("http://127.0.0.1:8188/queue", timeout_seconds=180):
            raise RuntimeError("ComfyUI server did not become ready in time")

        if not Path(WORKFLOW_JSON).exists():
            raise FileNotFoundError(f"Workflow JSON not found at {WORKFLOW_JSON}. Export one from ComfyUI (File -> Export API)")

        prompt_obj = json.loads(Path(WORKFLOW_JSON).read_text())
        
        # Inject user inputs into the workflow
        # Update text prompt (node 6 - CLIP Text Encode positive prompt)
        if "6" in prompt_obj and "inputs" in prompt_obj["6"]:
            # 🎯 PROMPT TUNING: Enhance user prompt here
            enhanced_prompt = _enhance_prompt(prompt)
            prompt_obj["6"]["inputs"]["text"] = enhanced_prompt
            print(f"Original prompt: {prompt}")
            print(f"Enhanced prompt: {enhanced_prompt}")
            
        # Update reference image path (node 52 - Load Image)  
        if "52" in prompt_obj and "inputs" in prompt_obj["52"]:
            image_filename = Path(ref_image_path).name
            prompt_obj["52"]["inputs"]["image"] = image_filename
            print(f"Updated image in node 52: {image_filename}")
            
        # Update audio file path (node 58 - LoadAudio)
        if "58" in prompt_obj and "inputs" in prompt_obj["58"]:
            audio_filename = Path(audio_path).name
            prompt_obj["58"]["inputs"]["audio"] = audio_filename
            print(f"Updated audio in node 58: {audio_filename}")
            
        # Update audio file path (node 63 - LoadAudio for reference)
        if "63" in prompt_obj and "inputs" in prompt_obj["63"]:
            audio_filename = Path(audio_path).name
            prompt_obj["63"]["inputs"]["audio"] = audio_filename
            print(f"Updated reference audio in node 63: {audio_filename}")

        print("Submitting workflow to ComfyUI...")
        _post_prompt(prompt_obj)

        print("Waiting for outputs...")
        output_dir = Path(f"{COMFY_REMOTE_DIR}/output")
        Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)

        wait_start = time.time()
        latest_before = _latest_file_in_dir(output_dir)
        while time.time() - wait_start < 1800:
            # Check for a real video first
            video = _latest_video_file_in_dir(output_dir)
            if video and (latest_before is None or video != latest_before or video.stat().st_mtime > latest_before.stat().st_mtime):
                if _wait_until_stable(video):
                    Path(OUT_PATH).write_bytes(video.read_bytes())
                    print(f"Output saved to: {OUT_PATH} (copied from {video})")
                    return OUT_PATH
            # Otherwise, look for a frame sequence and encode it to video
            frames = _collect_new_frames(output_dir, wait_start)
            if len(frames) >= 8:
                last_frame = frames[-1]
                if _wait_until_stable(last_frame, stable_checks=5, sleep_seconds=2.0):
                    if _encode_frames_to_video(frames, fps=25, out_path=Path(OUT_PATH)):
                        print(f"Output saved to: {OUT_PATH} (encoded from {len(frames)} frames)")
                        return OUT_PATH
            time.sleep(2)
            time.sleep(2)

        raise RuntimeError("Timed out waiting for ComfyUI output")

    finally:
        # Keep server running for subsequent calls (do not terminate)
        pass


@app.function(
    image=image,
    gpu="H200",
    volumes={
        "/wan22": WEIGHTS_VOL.read_only(),
        "/cache": CACHE_VOL,
    },
    secrets=[modal.Secret.from_name("openai-secret")],
    timeout=60 * 60,
    min_containers=1,
    scaledown_window=3600,
)
def generate_video():
    return _generate_impl()


@app.function(
    image=image,
    gpu="H200",
    volumes={
        "/wan22": WEIGHTS_VOL.read_only(),
        "/cache": CACHE_VOL,
    },
    secrets=[modal.Secret.from_name("openai-secret")],
    timeout=60 * 60,
    min_containers=1,
    scaledown_window=3600,
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, Form, UploadFile, File, HTTPException
    from fastapi.responses import FileResponse
    
    web_app = FastAPI()
    
    @web_app.post("/")
    async def api_generate(
        prompt: str = Form(...),
        referenceImage: UploadFile = File(...),
        audioFile: UploadFile = File(...)
    ):
        try:
            print(f"Received request - Prompt: {prompt}, Image: {referenceImage.filename}, Audio: {audioFile.filename}")
            
            # Save uploaded files to writable cache directory
            ref_image_path = "/cache/user_input_image.jpg"
            audio_path = "/cache/user_input_audio.mp3"
            
            # Save reference image
            Path(ref_image_path).parent.mkdir(parents=True, exist_ok=True)
            with open(ref_image_path, "wb") as f:
                content = await referenceImage.read()
                f.write(content)
            
            # Save audio file  
            Path(audio_path).parent.mkdir(parents=True, exist_ok=True)
            with open(audio_path, "wb") as f:
                content = await audioFile.read()
                f.write(content)
            
            print("Files saved successfully")
            
            # Generate video with user inputs
            out_path = _generate_impl(
                prompt=prompt,
                ref_image_path=ref_image_path,
                audio_path=audio_path
            )
            
            print(f"Video generated: {out_path}")
            return FileResponse(path=out_path, media_type="video/mp4", filename="out.mp4")
            
        except Exception as e:
            print(f"Error in api_generate: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))
    
    return web_app


def _sync_local_repo_into_volume():
    """Upload local Wan repo directories into the weights volume using batch_upload."""
    src = Path(WAN_LOCAL_DIR)
    print(f"Syncing contents of {src} -> weights volume ...")
    with WEIGHTS_VOL.batch_upload() as batch:
        # Upload model shards and assets expected by the YAML paths
        for folder in ["split_files", "examples", "workflows"]:
            local_dir = src / folder
            if local_dir.exists():
                batch.put_directory(str(local_dir), f"/{folder}")
                print(f"Uploaded {local_dir} -> /{folder}")
            else:
                print(f"Skipping missing folder: {local_dir}")
    print("Sync complete.")

@app.local_entrypoint()
def main():
    if SYNC_WEIGHTS:
        _sync_local_repo_into_volume()
    print(generate_video.remote())

# modal volume rm weights_vol /workflows/Wan2.2Sound2Vid.json
# modal volume put "weights_vol" "./Wan_2.2_ComfyUI_Repackaged/workflows/Wan2.2Sound2Vid.json" /workflows/Wan2.2Sound2Vid.json -f
# modal volume ls "weights_vol" /examples
# modal volume ls "weights_vol" /workflows

#modal volume get wan22-s2v-cache /out.mp4 ./out.mp4

#music video of a girl on a surfboard, sunset lighting, cinematic, music video

#dynamic music video of a handsome man, polaroid, nostalgia, sad, retro, VHS, heartbreak, singing