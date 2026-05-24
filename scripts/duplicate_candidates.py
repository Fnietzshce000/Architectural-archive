"""Find duplicate/near-duplicate image candidates with optional imagededup.

This script only writes a report. It never deletes duplicate files.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_settings

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
HASH_METHODS = {"phash", "dhash", "ahash", "whash"}
ALL_METHODS = HASH_METHODS | {"cnn"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=json_safe)
    temp_path.replace(path)


def collect_images(images_dir: Path, *, sample: int, use_all: bool, seed: int) -> List[Path]:
    if use_all or sample <= 0:
        return sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)

    files: List[Path] = []
    for path in images_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            files.append(path)
            if len(files) >= sample:
                break
    return sorted(files)


def load_method(method_name: str):
    try:
        from imagededup.methods import AHash, CNN, DHash, PHash, WHash
    except ImportError as exc:
        raise SystemExit(
            "imagededup yuklu degil. Kurulum: pip install -r requirements.optional.txt "
            "veya pip install imagededup"
        ) from exc

    mapping = {
        "phash": PHash,
        "dhash": DHash,
        "ahash": AHash,
        "whash": WHash,
        "cnn": CNN,
    }
    return mapping[method_name](verbose=True)


def normalize_duplicate_map(duplicate_map: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Set[str]]]:
    edges = []
    adjacency: Dict[str, Set[str]] = {}
    seen_pairs = set()

    for source, values in duplicate_map.items():
        adjacency.setdefault(source, set())
        for value in values or []:
            score = None
            target = value
            if isinstance(value, (list, tuple)) and value:
                target = value[0]
                if len(value) > 1:
                    score = value[1]
            source = str(source)
            target = str(target)
            if not target or target == source:
                continue
            pair = tuple(sorted((source, target)))
            adjacency.setdefault(source, set()).add(target)
            adjacency.setdefault(target, set()).add(source)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            edge = {"source": source, "target": target}
            if score is not None:
                edge["score"] = json_safe(score)
            edges.append(edge)
    return edges, adjacency


def connected_groups(adjacency: Dict[str, Set[str]]) -> List[List[str]]:
    visited: Set[str] = set()
    groups: List[List[str]] = []
    for node in sorted(adjacency):
        if node in visited or not adjacency[node]:
            continue
        stack = [node]
        group = []
        visited.add(node)
        while stack:
            current = stack.pop()
            group.append(current)
            for nxt in adjacency.get(current, set()):
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        if len(group) > 1:
            groups.append(sorted(group))
    return groups


def build_report(args: argparse.Namespace) -> Dict[str, Any]:
    if args.method not in ALL_METHODS:
        raise SystemExit(f"Desteklenmeyen method: {args.method}. Secenekler: {sorted(ALL_METHODS)}")

    settings = get_settings()
    images_dir = Path(args.images_dir) if args.images_dir else settings.get_images_path()
    output = Path(args.output) if args.output else settings.get_data_path() / "reports" / "duplicates_candidates.json"
    files = collect_images(images_dir, sample=args.sample, use_all=args.all, seed=args.seed)

    if args.dry_run:
        return {
            "generated_at": utc_now(),
            "tool": "imagededup",
            "dry_run": True,
            "method": args.method,
            "images_dir": str(images_dir),
            "output": str(output),
            "planned_images": len(files),
            "sample": args.sample,
            "all": args.all,
            "threshold": args.threshold,
        }

    if not files:
        raise SystemExit(f"Gorsel bulunamadi: {images_dir}")

    method = load_method(args.method)
    encodings = {}
    for index, image_path in enumerate(files, start=1):
        try:
            encodings[image_path.name] = method.encode_image(image_file=str(image_path))
        except Exception as exc:
            print(f"Encode atlandi: {image_path} - {exc}")
        if args.progress and index % args.progress == 0:
            print(f"Encoded {index:,}/{len(files):,}")

    threshold_arg: Dict[str, Any]
    if args.method == "cnn":
        threshold_arg = {"min_similarity_threshold": args.threshold}
    else:
        threshold_arg = {"max_distance_threshold": int(args.threshold)}

    duplicate_map = method.find_duplicates(encoding_map=encodings, scores=True, **threshold_arg)
    edges, adjacency = normalize_duplicate_map(duplicate_map)
    groups = connected_groups(adjacency)
    group_records = [
        {
            "group_id": idx + 1,
            "count": len(files_in_group),
            "files": files_in_group,
            "review_candidates": files_in_group[1:],
        }
        for idx, files_in_group in enumerate(groups[: args.max_groups])
    ]

    report = {
        "generated_at": utc_now(),
        "tool": "imagededup",
        "dry_run": False,
        "method": args.method,
        "images_dir": str(images_dir),
        "output": str(output),
        "checked_images": len(files),
        "encoded_images": len(encodings),
        "sample": args.sample,
        "all": args.all,
        "threshold": args.threshold,
        "pair_count": len(edges),
        "group_count": len(groups),
        "edges_preview": edges[: args.max_edges],
        "groups": group_records,
    }
    atomic_write_json(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="imagededup ile duplicate aday raporu uretir; dosya silmez.")
    parser.add_argument("--images-dir", default="", help="Varsayilan: settings data/images")
    parser.add_argument("--output", default="", help="Varsayilan: data/reports/duplicates_candidates.json")
    parser.add_argument("--method", default="phash", choices=sorted(ALL_METHODS))
    parser.add_argument("--threshold", type=float, default=10, help="Hash icin max distance, cnn icin min similarity.")
    parser.add_argument("--sample", type=int, default=10000, help="Varsayilan orneklem. 0 veya --all tum dosyalari kullanir.")
    parser.add_argument("--all", action="store_true", help="Tum gorselleri tara. Buyuk arsivlerde uzun surebilir.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-groups", type=int, default=200)
    parser.add_argument("--max-edges", type=int, default=1000)
    parser.add_argument("--progress", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true", help="Sadece plan yazdirir; imagededup calismaz, rapor yazmaz.")
    args = parser.parse_args()

    report = build_report(args)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=json_safe))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
