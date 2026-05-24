"""Run an optional CleanVision image quality audit and write a safe JSON report.

This script never deletes, moves, or modifies archive images. By default it checks a
sample and writes data/reports/image_quality_report.json unless --dry-run is used.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_settings

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_ISSUE_TYPES = [
    "dark",
    "light",
    "blurry",
    "low_information",
    "odd_size",
    "odd_aspect_ratio",
    "exact_duplicates",
    "near_duplicates",
]


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
    if isinstance(value, float):
        if value != value:
            return None
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


def parse_issue_types(value: str) -> List[str]:
    if not value.strip():
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def dataframe_records(frame: Any) -> List[Dict[str, Any]]:
    if frame is None:
        return []
    try:
        return [
            {str(key): json_safe(value) for key, value in row.items()}
            for row in frame.to_dict(orient="records")
        ]
    except Exception:
        return []


def top_issue_examples(issues: Any, *, top: int) -> Dict[str, List[Dict[str, Any]]]:
    examples: Dict[str, List[Dict[str, Any]]] = {}
    if issues is None or top <= 0:
        return examples

    for column in list(getattr(issues, "columns", [])):
        if not (column.startswith("is_") and column.endswith("_issue")):
            continue
        issue_type = column[3:-6]
        try:
            rows = issues[issues[column] == True].copy()  # noqa: E712 - pandas bool mask
            if rows.empty:
                continue
            score_col = f"{issue_type}_score"
            if score_col in rows.columns:
                rows = rows.sort_values(score_col, ascending=True)
            items = []
            for index, row in rows.head(top).iterrows():
                item = {"image_path": str(index)}
                if score_col in row:
                    item["score"] = json_safe(row[score_col])
                items.append(item)
            examples[issue_type] = items
        except Exception as exc:
            examples[issue_type] = [{"image_path": "error", "score": str(exc)}]
    return examples


def build_report(args: argparse.Namespace) -> Dict[str, Any]:
    settings = get_settings()
    images_dir = Path(args.images_dir) if args.images_dir else settings.get_images_path()
    output = Path(args.output) if args.output else settings.get_data_path() / "reports" / "image_quality_report.json"
    files = collect_images(images_dir, sample=args.sample, use_all=args.all, seed=args.seed)

    if args.dry_run:
        return {
            "generated_at": utc_now(),
            "tool": "cleanvision",
            "dry_run": True,
            "images_dir": str(images_dir),
            "output": str(output),
            "planned_images": len(files),
            "sample": args.sample,
            "all": args.all,
        }

    try:
        from cleanvision import Imagelab
    except ImportError as exc:
        raise SystemExit(
            "CleanVision yuklu degil. Kurulum: pip install -r requirements.optional.txt "
            "veya pip install cleanvision"
        ) from exc

    if not files:
        raise SystemExit(f"Gorsel bulunamadi: {images_dir}")

    possible = set(Imagelab.list_possible_issue_types())
    requested = parse_issue_types(args.issue_types) or DEFAULT_ISSUE_TYPES
    issue_types = {name: {} for name in requested if name in possible}
    skipped = [name for name in requested if name not in possible]

    imagelab = Imagelab(filepaths=[str(path) for path in files], verbose=not args.quiet)
    if issue_types:
        imagelab.find_issues(issue_types=issue_types, n_jobs=args.jobs, verbose=not args.quiet)
    else:
        imagelab.find_issues(n_jobs=args.jobs, verbose=not args.quiet)

    summary_records = dataframe_records(getattr(imagelab, "issue_summary", None))
    report = {
        "generated_at": utc_now(),
        "tool": "cleanvision",
        "dry_run": False,
        "images_dir": str(images_dir),
        "output": str(output),
        "checked_images": len(files),
        "sample": args.sample,
        "all": args.all,
        "issue_types": list(issue_types.keys()) if issue_types else "cleanvision_default",
        "skipped_issue_types": skipped,
        "issue_summary": summary_records,
        "top_examples": top_issue_examples(getattr(imagelab, "issues", None), top=args.top),
    }
    atomic_write_json(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="CleanVision ile gorsel kalite raporu uretir; veri silmez.")
    parser.add_argument("--images-dir", default="", help="Varsayilan: settings data/images")
    parser.add_argument("--output", default="", help="Varsayilan: data/reports/image_quality_report.json")
    parser.add_argument("--sample", type=int, default=5000, help="Varsayilan orneklem. 0 veya --all tum dosyalari kullanir.")
    parser.add_argument("--all", action="store_true", help="Tum gorselleri tara. Buyuk arsivlerde uzun surebilir.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--issue-types", default=",".join(DEFAULT_ISSUE_TYPES), help="Virgulle ayrilmis CleanVision issue type listesi.")
    parser.add_argument("--jobs", type=int, default=1, help="Windows icin 1 daha stabil olabilir.")
    parser.add_argument("--top", type=int, default=25, help="Her issue icin rapora yazilacak en agir ornek sayisi.")
    parser.add_argument("--dry-run", action="store_true", help="Sadece plan yazdirir; CleanVision calismaz, rapor yazmaz.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    report = build_report(args)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=json_safe))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
