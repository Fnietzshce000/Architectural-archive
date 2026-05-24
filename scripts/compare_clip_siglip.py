"""
Compare the current OpenCLIP model against SigLIP/SigLIP 2 on the same image sample.

Read-only for the archive:
- samples images from data/images
- encodes the same sample with both models
- writes JSON plus contact sheets under data/reports
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

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
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


def iter_images(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def reservoir_sample(paths: Iterable[Path], sample_size: int, seed: int, max_scan: int | None) -> List[Path]:
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


def load_queries(query_file: str | None) -> List[str]:
    if not query_file:
        return DEFAULT_QUERIES
    path = Path(query_file)
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def normalize(features: torch.Tensor) -> torch.Tensor:
    return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def get_default_images_dir() -> Path:
    try:
        from config import get_settings

        return get_settings().get_images_path()
    except Exception:
        return ROOT / "data" / "images"


def encode_openclip_images(
    model,
    preprocess,
    image_paths: List[Path],
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, List[Path]]:
    vectors = []
    valid_paths = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        tensors = []
        loaded_paths = []
        for path in batch_paths:
            try:
                tensors.append(preprocess(Image.open(path).convert("RGB")))
                loaded_paths.append(path)
            except Exception:
                continue
        if not tensors:
            continue
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            features = model.encode_image(batch)
        vectors.append(normalize(features).float().cpu().numpy())
        valid_paths.extend(loaded_paths)
    return np.vstack(vectors), valid_paths


def encode_openclip_texts(model, tokenizer, queries: List[str], device: str) -> np.ndarray:
    tokens = tokenizer(queries).to(device)
    with torch.no_grad():
        features = model.encode_text(tokens)
    return normalize(features).float().cpu().numpy()


def get_siglip_tensor(output, preferred_attr: str):
    if hasattr(output, "norm"):
        return output
    for attr in (preferred_attr, "image_embeds", "text_embeds", "pooler_output"):
        value = getattr(output, attr, None)
        if value is not None:
            return value
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    raise TypeError(f"Could not extract feature tensor from {type(output)!r}")


def encode_siglip_images(model, processor, image_paths: List[Path], batch_size: int, device: str) -> tuple[np.ndarray, List[Path]]:
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
            output = model(**inputs)
        vectors.append(normalize(get_siglip_tensor(output, "image_embeds")).float().cpu().numpy())
        valid_paths.extend(loaded_paths)
    return np.vstack(vectors), valid_paths


def encode_siglip_texts(model, processor, queries: List[str], prompt_template: str, device: str) -> np.ndarray:
    dummy_image = Image.new("RGB", (224, 224), color=(255, 255, 255))
    texts = [prompt_template.format(query) for query in queries]
    inputs = processor(text=texts, images=dummy_image, padding="max_length", return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**inputs)
    return normalize(get_siglip_tensor(output, "text_embeds")).float().cpu().numpy()


def build_results(queries: List[str], text_vectors: np.ndarray, image_vectors: np.ndarray, image_paths: List[Path], top_k: int) -> dict:
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


def make_contact_sheet(openclip_results: dict, siglip_results: dict, output_dir: Path, top_n: int = 5) -> List[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    written = []
    thumb_w, thumb_h = 150, 150
    label_h = 44
    left_w = 180
    width = left_w + top_n * thumb_w
    row_h = thumb_h + label_h

    for query in openclip_results:
        rows = [("OpenCLIP", openclip_results[query][:top_n]), ("SigLIP2", siglip_results[query][:top_n])]
        canvas = Image.new("RGB", (width, row_h * 2 + 36), color=(245, 245, 245))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), query, fill=(20, 20, 20), font=font)
        for row_index, (model_name, items) in enumerate(rows):
            y = 36 + row_index * row_h
            draw.text((8, y + 8), model_name, fill=(20, 20, 20), font=font)
            for col, item in enumerate(items):
                x = left_w + col * thumb_w
                try:
                    img = Image.open(item["image_path"]).convert("RGB")
                    img.thumbnail((thumb_w, thumb_h))
                    px = x + (thumb_w - img.width) // 2
                    py = y + (thumb_h - img.height) // 2
                    canvas.paste(img, (px, py))
                except Exception:
                    draw.rectangle((x, y, x + thumb_w - 1, y + thumb_h - 1), outline=(180, 0, 0))
                draw.text((x + 4, y + thumb_h + 2), f"#{item['rank']} {item['score']:.3f}", fill=(20, 20, 20), font=font)
                draw.text((x + 4, y + thumb_h + 18), item["image_name"][:22], fill=(60, 60, 60), font=font)
        safe_name = "".join(ch if ch.isalnum() else "_" for ch in query.lower()).strip("_")
        out_path = output_dir / f"{safe_name}.jpg"
        canvas.save(out_path, quality=92)
        written.append(str(out_path))
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare OpenCLIP vs SigLIP 2 on the same image sample.")
    parser.add_argument("--sample", type=int, default=300)
    parser.add_argument("--max-scan", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--query-file", default=None)
    parser.add_argument("--output", default="data/reports/clip_vs_siglip2_compare.json")
    parser.add_argument("--sheets-dir", default="data/reports/clip_vs_siglip2_sheets")
    parser.add_argument("--openclip-model", default="ViT-L-14")
    parser.add_argument("--openclip-pretrained", default="laion2b_s32b_b82k")
    parser.add_argument("--siglip-model", default="google/siglip2-base-patch16-224")
    parser.add_argument("--siglip-prompt-template", default="This is a photo of {}.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    images_dir = Path(args.images_dir) if args.images_dir else get_default_images_dir()
    output_path = Path(args.output)
    sheets_dir = Path(args.sheets_dir)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    if not sheets_dir.is_absolute():
        sheets_dir = ROOT / sheets_dir

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    queries = load_queries(args.query_file)
    image_paths = reservoir_sample(iter_images(images_dir), args.sample, args.seed, args.max_scan)
    print(f"Sampled {len(image_paths)} images from {images_dir}")
    print(f"Device: {device}")

    started = time.time()

    import open_clip

    print(f"Loading OpenCLIP: {args.openclip_model} / {args.openclip_pretrained}")
    oc_model, _, oc_preprocess = open_clip.create_model_and_transforms(
        args.openclip_model,
        pretrained=args.openclip_pretrained,
    )
    oc_tokenizer = open_clip.get_tokenizer(args.openclip_model)
    oc_model = oc_model.to(device).eval()
    oc_image_vectors, valid_paths = encode_openclip_images(oc_model, oc_preprocess, image_paths, args.batch_size, device)
    oc_text_vectors = encode_openclip_texts(oc_model, oc_tokenizer, queries, device)
    openclip_results = build_results(queries, oc_text_vectors, oc_image_vectors, valid_paths, args.top_k)

    del oc_model

    from transformers import AutoModel, AutoProcessor

    print(f"Loading SigLIP: {args.siglip_model}")
    sl_model = AutoModel.from_pretrained(args.siglip_model, attn_implementation="sdpa", low_cpu_mem_usage=True).to(device).eval()
    sl_processor = AutoProcessor.from_pretrained(args.siglip_model)
    sl_image_vectors, sl_valid_paths = encode_siglip_images(sl_model, sl_processor, valid_paths, args.batch_size, device)
    sl_text_vectors = encode_siglip_texts(sl_model, sl_processor, queries, args.siglip_prompt_template, device)
    siglip_results = build_results(queries, sl_text_vectors, sl_image_vectors, sl_valid_paths, args.top_k)

    sheet_paths = make_contact_sheet(openclip_results, siglip_results, sheets_dir, top_n=min(5, args.top_k))

    overlaps = {}
    for query in queries:
        oc_top = [row["image_name"] for row in openclip_results[query][:5]]
        sl_top = [row["image_name"] for row in siglip_results[query][:5]]
        overlaps[query] = {
            "top1_same": bool(oc_top and sl_top and oc_top[0] == sl_top[0]),
            "top5_overlap": len(set(oc_top) & set(sl_top)),
        }

    report = {
        "openclip": {"model": args.openclip_model, "pretrained": args.openclip_pretrained},
        "siglip": {"model": args.siglip_model, "prompt_template": args.siglip_prompt_template},
        "device": device,
        "sample_requested": args.sample,
        "sample_encoded": len(valid_paths),
        "queries": queries,
        "elapsed_seconds": round(time.time() - started, 3),
        "overlap_summary": overlaps,
        "contact_sheets": sheet_paths,
        "openclip_results": openclip_results,
        "siglip_results": siglip_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote report: {output_path}")
    print(f"Wrote sheets: {sheets_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

