import json
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image
import hashlib
import random
import re
import csv

from tqdm.auto import tqdm

COCO_URLS = {
    "images": "http://images.cocodataset.org/zips/{split}.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


def _download(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))


def download_coco(coco_dir: Path, split: str = "val2017", keep_zips: bool = False) -> None:
    coco_dir.mkdir(parents=True, exist_ok=True)

    captions_file = coco_dir / "annotations" / f"captions_{split}.json"
    if not captions_file.exists():
        zip_path = coco_dir / "annotations_trainval2017.zip"
        if not zip_path.exists():
            _download(COCO_URLS["annotations"], zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract(f"annotations/captions_{split}.json", coco_dir)
        if not keep_zips:
            zip_path.unlink()
    print("Captions:", captions_file)

    image_dir = coco_dir / split
    if not image_dir.is_dir() or not any(image_dir.iterdir()):
        zip_path = coco_dir / f"{split}.zip"
        if not zip_path.exists():
            _download(COCO_URLS["images"].format(split=split), zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(coco_dir)
        if not keep_zips:
            zip_path.unlink()
    print("Images  :", image_dir)


def load_coco_captions(coco_dir: Path, split: str = "val2017"):
    data = json.loads((coco_dir / "annotations" / f"captions_{split}.json").read_text())

    images = {
        img["id"]: {"file_name": img["file_name"], "width": img["width"], "height": img["height"]}
        for img in data["images"]
    }
    captions = defaultdict(list)
    for ann in data["annotations"]:
        captions[ann["image_id"]].append(ann["caption"])
    return images, dict(captions)
    

def normalize_caption(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip().lower())
    return text.rstrip(" .")


def filter_images(images: dict, image_dir: Path, min_short_side: int, max_aspect: float):
    kept, dropped, seen = {}, Counter(), set()
    for image_id, meta in tqdm(images.items(), desc="Checking images"):
        w, h = meta["width"], meta["height"]
        if min(w, h) < min_short_side:
            dropped["too_small"] += 1
            continue
        if max(w, h) / min(w, h) > max_aspect:
            dropped["extreme_aspect_ratio"] += 1
            continue

        path = image_dir / meta["file_name"]
        with Image.open(path) as img:          # reads the header only
            mode = img.mode
        if mode != "RGB":
            dropped["not_rgb"] += 1
            continue

        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest in seen:
            dropped["duplicate_image"] += 1
            continue
        seen.add(digest)
        kept[image_id] = meta
    return kept, dropped


def select_captions(image_ids, captions: dict, tokenizer, max_tokens: int = 77, seed: int = 42):
    rng = random.Random(seed)
    chosen, dropped, used = {}, Counter(), set()

    for image_id in sorted(image_ids):
        valid = []
        for cap in dict.fromkeys(normalize_caption(c) for c in captions[image_id]):
            if len(tokenizer(cap).input_ids) > max_tokens:   # includes start/end tokens
                dropped["captions_over_token_limit"] += 1
            else:
                valid.append(cap)

        rng.shuffle(valid)
        pick = next((c for c in valid if c not in used), None)
        if pick is None:
            dropped["images_without_unique_caption"] += 1
            continue
        used.add(pick)
        chosen[image_id] = pick
    return chosen, dropped


def preprocess_image(src: Path, dest: Path, size: int = 512) -> None:
    with Image.open(src) as img:
        img = img.convert("RGB")
        scale = size / min(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)),
                         Image.Resampling.LANCZOS)
        left, top = (img.width - size) // 2, (img.height - size) // 2
        img.crop((left, top, left + size, top + size)).save(dest, quality=95)


def build_subset(chosen: dict, images: dict, image_dir: Path, out_dir: Path,
                 n: int, size: int = 512, seed: int = 42) -> list:
    rng = random.Random(seed)
    ids = sorted(rng.sample(sorted(chosen), n))

    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for image_id in tqdm(ids, desc="Preprocessing"):
        meta = images[image_id]
        file = f"images/{image_id:012d}.jpg"
        preprocess_image(image_dir / meta["file_name"], out_dir / file, size)
        rows.append({
            "image_id": image_id,
            "file": file,
            "caption": chosen[image_id],
            "coco_file_name": meta["file_name"],
            "orig_width": meta["width"],
            "orig_height": meta["height"],
        })

    with open(out_dir / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows
