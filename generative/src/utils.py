import platform
from pathlib import Path
import datetime
import json
from PIL import Image
from PIL import ImageDraw, ImageFont
import torch

# src/utils.py -> repo root is one level up from src/
REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = REPO_ROOT / "outputs"


def milestone_dir(name: str) -> Path:
    path = OUTPUTS_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_device():

    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def environment_info(device: str, dtype) -> dict:

    import diffusers
    import transformers

    return {
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "dtype": str(dtype),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "diffusers": diffusers.__version__,
            "transformers": transformers.__version__,
        },
    }

def make_generator(seed: int, device: str) -> torch.Generator:

    gen_device = "cuda" if device == "cuda" else "cpu"
    return torch.Generator(device=gen_device).manual_seed(seed)


def make_grid(images, cols: int) -> Image.Image:
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    grid = Image.new("RGB", (cols * w, rows * h), "white")
    for i, img in enumerate(images):
        grid.paste(img, ((i % cols) * w, (i // cols) * h))
    return grid


def save_run_log(out_dir: Path, env: dict, runs: dict, **config) -> Path:
    log = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **config,
        **env,
        "num_runs": len(runs),
        "runs": list(runs.values()),
    }
    path = out_dir / "run_log.json"
    path.write_text(json.dumps(log, indent=2))
    return path


def write_gallery(out_dir: Path, runs: dict, title: str) -> Path:
    lines = [f"# {title}", ""]
    experiments = {}
    for r in runs.values():
        experiments.setdefault(r["experiment"], []).append(r)

    for exp, entries in experiments.items():
        lines += [f"## {exp}", ""]
        for r in entries:
            lines += [
                f"### {r['tag']}",
                f"**Prompt:** {r['prompt']}  ",
                f"Scheduler: {r['scheduler']} | Steps: {r['steps']} | "
                f"Guidance: {r['guidance_scale']} | Seed: {r['seed']}",
                "",
                f"![{r['tag']}]({r['image']})",
                "",
            ]
    path = out_dir / "GALLERY.md"
    path.write_text("\n".join(lines))
    return path
    
    
def add_caption(img: Image.Image, text: str, height: int = 40) -> Image.Image:
    out = Image.new("RGB", (img.width, img.height + height), "white")
    out.paste(img, (0, height))
    font = ImageFont.load_default(size=height // 2)
    ImageDraw.Draw(out).text((10, height // 4), text, fill="black", font=font)
    return out
