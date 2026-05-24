"""
Small SigLIP 2 retrieval benchmark for the local image archive.

This script is intentionally read-only for the archive/index:
- reads images from data/images
- loads a SigLIP/SigLIP 2 model from Hugging Face
- writes a JSON report under data/reports
- does not touch ChromaDB or MegaSync state
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_QUERIES = [
    "modern black leather chair",
    "wood dining table",
    "classic sofa living room",
    "gold chandelier luxury lobby",
    "minimalist bedroom bed",
    "marble coffee table",
    "office desk chair",
    "outdoor garden furniture",
    "kitchen cabinet",
    "decorative wall mirror",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def reservoir_sample(paths: Iterable[Path], sample_size: int, seed: int, max_scan: int | None) -> List[Path]:
    """Sample without holding the whole image list in memory."""
    rng = random.Random(seed)
    sample: List[Path] = []
    seen = 0

    for path in paths:
        if max_scan is not None and seen >= max_scan:
            break
        seen += 1
        if len(sample) < sample_size:
            sample.append(path)
            continue
        replace_at = rng.randint(0, seen - 1)
        if replace_at < sample_size:
            sample[replace_at] = path

    return sample


def iter_images(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def load_queries(query_file: str | None) -> List[str]:
    if not query_file:
        return DEFAULT_QUERIES

    path = Path(query_file)
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def normalize(features):
    return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

def get_feature_tensor(output, preferred_attr: str):
    """Handle Transformers versions that return tensors or model output objects."""
    if hasattr(output, "norm"):
        return output
    for attr in (preferred_attr, "image_embeds", "text_embeds", "pooler_output"):
        value = getattr(output, attr, None)
        if value is not None:
            return value
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    raise TypeError(f"Could not extract feature tensor from {type(output)!r}")



def encode_images(model, processor, image_paths: List[Path], batch_size: int, device: str) -> tuple[np.ndarray, List[Path]]:
    import numpy as np
    import torch
    from PIL import Image

    vectors = []
    valid_paths = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        images = []
        loaded_paths = []

        for path in batch_paths:
            try:
                images.append(Image.open(path).convert("RGB"))
                loaded_paths.append(path)
            except Exception:
                continue

        if not images:
            continue

        inputs = processor(text=["This is a photo."], images=images, padding="max_length", return_tensors="pt").to(device)
        with torch.no_grad():
            if device == "cuda":
                with torch.amp.autocast("cuda"):
                    features = model(**inputs)
            else:
                features = model(**inputs)

        vectors.append(normalize(get_feature_tensor(features, "image_embeds")).float().cpu().numpy())
        valid_paths.extend(loaded_paths)

    if not vectors:
        return np.empty((0, 0), dtype=np.float32), []

    return np.vstack(vectors), valid_paths


def encode_texts(model, processor, queries: List[str], prompt_template: str, device: str) -> np.ndarray:
    import torch
    from PIL import Image

    texts = [prompt_template.format(query) for query in queries]
    dummy_image = Image.new("RGB", (224, 224), color=(255, 255, 255))
    inputs = processor(text=texts, images=dummy_image, padding="max_length", return_tensors="pt").to(device)

    with torch.no_grad():
        if device == "cuda":
            with torch.amp.autocast("cuda"):
                features = model(**inputs)
        else:
            features = model(**inputs)

    return normalize(get_feature_tensor(features, "text_embeds")).float().cpu().numpy()


def build_results(
    queries: List[str],
    text_vectors: np.ndarray,
    image_vectors: np.ndarray,
    image_paths: List[Path],
    top_k: int,
) -> dict:
    import numpy as np

    scores = text_vectors @ image_vectors.T
    results = {}

    for query_index, query in enumerate(queries):
        query_scores = scores[query_index]
        top_indices = np.argsort(-query_scores)[:top_k]
        results[query] = [
            {
                "rank": rank + 1,
                "score": round(float(query_scores[index]), 6),
                "image_path": str(image_paths[index]),
                "image_name": image_paths[index].name,
            }
            for rank, index in enumerate(top_indices)
        ]

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark SigLIP 2 on a safe sample of archive images.")
    parser.add_argument("--model", default="google/siglip2-base-patch16-224")
    parser.add_argument("--sample", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-scan", type=int, default=None)
    parser.add_argument("--query-file", default=None)
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--output", default="data/reports/siglip2_benchmark.json")
    parser.add_argument("--prompt-template", default="This is a photo of {}.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from config import get_settings
        default_images_dir = get_settings().get_images_path()
    except Exception:
        default_images_dir = ROOT / 'data' / 'images'
    images_dir = Path(args.images_dir) if args.images_dir else default_images_dir
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    queries = load_queries(args.query_file)
    image_paths = reservoir_sample(
        iter_images(images_dir),
        sample_size=args.sample,
        seed=args.seed,
        max_scan=args.max_scan,
    )

    print(f"Model: {args.model}")
    print(f"Images dir: {images_dir}")
    print(f"Sampled images: {len(image_paths)}")
    print(f"Queries: {len(queries)}")
    print(f"Output: {output_path}")

    if args.dry_run:
        return 0

    from transformers import AutoModel, AutoProcessor
    import torch

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    started = time.time()

    model = AutoModel.from_pretrained(args.model, attn_implementation="sdpa", low_cpu_mem_usage=True).to(device)
    processor = AutoProcessor.from_pretrained(args.model)
    model.eval()

    image_vectors, valid_paths = encode_images(
        model=model,
        processor=processor,
        image_paths=image_paths,
        batch_size=args.batch_size,
        device=device,
    )
    if image_vectors.size == 0:
        raise RuntimeError("No valid image vectors were produced.")

    text_vectors = encode_texts(
        model=model,
        processor=processor,
        queries=queries,
        prompt_template=args.prompt_template,
        device=device,
    )
    results = build_results(
        queries=queries,
        text_vectors=text_vectors,
        image_vectors=image_vectors,
        image_paths=valid_paths,
        top_k=args.top_k,
    )

    report = {
        "model": args.model,
        "device": device,
        "sample_requested": args.sample,
        "sample_encoded": len(valid_paths),
        "query_count": len(queries),
        "image_dim": int(image_vectors.shape[1]),
        "elapsed_seconds": round(time.time() - started, 3),
        "prompt_template": args.prompt_template,
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





