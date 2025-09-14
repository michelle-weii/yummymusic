import modal, subprocess, os
from pathlib import Path

app = modal.App("wan2.2-s2v-14b-yummymusic")

# HF_REPO = "Wan-AI/Wan2.2-S2V-14B"
# MODEL_DIR = "/models/Wan2.2-S2V-14B"
# WAN_REPO_DIR = "/wan22"
CACHE_VOL = modal.Volume.from_name("wan22-s2v-cache", create_if_missing=True)
WAN_LOCAL_DIR = "/Users/willi1/yummymusic/Wan_2.2_ComfyUI_Repackaged"
COMFY_LOCAL_DIR = "/Users/willi1/yummymusic/ComfyUI"

# Hard-coded inputs
PROMPT = "a white cat wearing sunglasses on a surfboard"
REF_IMAGE = "/wan22/examples/i2v_input.jpg"
AUDIO = "/wan22/examples/talk.wav"
OUT_PATH = "/models/out.mp4"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .uv_pip_install("torch", "torchvision", "torchaudio", "transformers", "diffusers", "huggingface_hub")
    .add_local_dir(COMFY_LOCAL_DIR)
    .run_commands(
        f"python -m pip install --upgrade pip",
        f"python -m pip install -r {COMFY_LOCAL_DIR}/requirements.txt",
    )
)

@app.function(
    image=image,
    gpu="H200",
    secrets=[modal.Secret.from_name("huggingface-token")],
    volumes={"/models": CACHE_VOL, "/wan22": CACHE_VOL},
    timeout=60 * 60,
)
def generate_video():
    """Run inference using ComfyUI and cached model artifacts."""

    # Download model if needed
    # if not Path(MODEL_DIR).exists() or not any(Path(MODEL_DIR).iterdir()):
    #     print(f"Downloading model from {HF_REPO}...")
    #     token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    #     # Download model files
    #     model = AutoModel.from_pretrained(
    #         HF_REPO,
    #         token=token,
    #         cache_dir=MODEL_DIR,
    #         torch_dtype=torch.float16,
    #         device_map="auto"
    #     )

    #     tokenizer = AutoTokenizer.from_pretrained(
    #         HF_REPO,
    #         token=token,
    #         cache_dir=MODEL_DIR
    #     )
    # else:
    #     # Load existing model
    #     model = AutoModel.from_pretrained(
    #         MODEL_DIR,
    #         torch_dtype=torch.float16,
    #         device_map="auto"
    #     )
    #     tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    print("Starting ComfyUI inference...")
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    # Example: call a ComfyUI script entrypoint. Adjust to your pipeline.
    # If your ComfyUI workflow is CLI-driven, run it here.
    # subprocess.run([...], check=True)
    # For now, create a placeholder file.
    Path(OUT_PATH).touch()

    print(f"Video saved to: {OUT_PATH}")
    return OUT_PATH


def _sync_local_repo_into_volume():
    """Upload local Wan repo folder into the persistent volume under /wan22."""
    # Upload contents of WAN_LOCAL_DIR into the root of the volume.
    # When mounted at /wan22, contents will be available directly under that path.
    src = Path(WAN_LOCAL_DIR)
    dst = "/"
    print(f"Syncing contents of {src} -> volume:{dst} ...")
    for child in src.iterdir():
        CACHE_VOL.put(str(child), remote_path=dst)
    print("Sync complete.")


if __name__ == "__main__":
    _sync_local_repo_into_volume()
    with app.run():
        result = generate_video.remote()
        print(f"Video generated at: {result}")

