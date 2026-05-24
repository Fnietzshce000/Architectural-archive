"""POC runner for Recognize Anything (RAM/RAM++/Tag2Text) tagging.

This script calls the official repository inference scripts on a small sample and
writes a JSON report. It does not modify ChromaDB or archive metadata.
"""
import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_settings

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MODEL_SCRIPTS = {
    "ram_plus": "inference_ram_plus.py",
    "ram": "inference_ram.py",
    "tag2text": "inference_tag2text.py",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    temp_path.replace(path)


def collect_images(images_dir: Path, limit: int) -> List[Path]:
    files: List[Path] = []
    for path in images_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            files.append(path)
            if len(files) >= max(1, limit):
                break
    return sorted(files)


def parse_tags(output: str) -> List[str]:
    for line in output.splitlines():
        if "Image Tags:" in line:
            raw = line.split("Image Tags:", 1)[1]
            return [tag.strip() for tag in re.split(r"\s*\|\s*|,", raw) if tag.strip()]
    return []


def build_install_hint(repo_path: Path) -> str:
    return (
        "Recognize Anything repo hazir degil. POC icin: "
        f"git clone https://github.com/xinyu1205/recognize-anything.git {repo_path} "
        "ve checkpoint dosyasini repo icindeki pretrained/ klasorune indir. "
        "Alternatif: pip install git+https://github.com/xinyu1205/recognize-anything.git"
    )


def run_poc(args: argparse.Namespace) -> Dict[str, Any]:
    settings = get_settings()
    images_dir = Path(args.images_dir) if args.images_dir else settings.get_images_path()
    repo_path = Path(args.repo_path)
    output = Path(args.output) if args.output else settings.get_data_path() / "reports" / "ram_tagging_report.json"
    images = collect_images(images_dir, args.sample)
    script_path = repo_path / MODEL_SCRIPTS[args.model]
    checkpoint = Path(args.checkpoint) if args.checkpoint else repo_path / "pretrained" / args.default_checkpoint

    plan = {
        "generated_at": utc_now(),
        "tool": "recognize-anything",
        "model": args.model,
        "dry_run": args.dry_run,
        "repo_path": str(repo_path),
        "script_path": str(script_path),
        "checkpoint": str(checkpoint),
        "images_dir": str(images_dir),
        "output": str(output),
        "planned_images": len(images),
    }

    if args.dry_run:
        return plan

    if not script_path.exists():
        raise SystemExit(build_install_hint(repo_path))
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint bulunamadi: {checkpoint}")

    records = []
    for image_path in images:
        command = [args.python, str(script_path), "--image", str(image_path), "--pretrained", str(checkpoint)]
        try:
            completed = subprocess.run(
                command,
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=args.timeout,
                check=False,
            )
            combined = (completed.stdout or "") + "\n" + (completed.stderr or "")
            records.append(
                {
                    "image_path": str(image_path),
                    "returncode": completed.returncode,
                    "tags": parse_tags(combined),
                    "output_preview": combined.strip()[-2000:],
                }
            )
        except Exception as exc:
            records.append({"image_path": str(image_path), "returncode": -1, "tags": [], "error": str(exc)})

    report = {
        **plan,
        "dry_run": False,
        "tagged_images": sum(1 for item in records if item.get("tags")),
        "records": records,
    }
    atomic_write_json(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Recognize Anything RAM/RAM++ tagging POC raporu uretir.")
    parser.add_argument("--images-dir", default="", help="Varsayilan: settings data/images")
    parser.add_argument("--repo-path", default=str(PROJECT_ROOT / "external" / "recognize-anything"))
    parser.add_argument("--checkpoint", default="", help="Checkpoint yolu. Bos ise repo/pretrained varsayilani kullanilir.")
    parser.add_argument("--default-checkpoint", default="ram_plus_swin_large_14m.pth")
    parser.add_argument("--model", default="ram_plus", choices=sorted(MODEL_SCRIPTS))
    parser.add_argument("--sample", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output", default="", help="Varsayilan: data/reports/ram_tagging_report.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report = run_poc(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
