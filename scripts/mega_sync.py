import argparse
import asyncio
import json
import logging
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from config import MetadataSchema, get_settings
from indexer.chroma_store import ChromaStore
from indexer.clip_encoder import encode_image, encode_text, load_model
from scraper.client import TelegramClientManager


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MegaSync")

DESIGN_TAXONOMY = [
    "modern", "classic", "minimalist", "scandinavian", "industrial",
    "vintage", "retro", "bohemian", "luxury", "rustic", "contemporary",
    "baroque", "art deco", "mid-century", "futuristic",
    "wood", "metal", "glass", "marble", "leather", "fabric", "plastic",
    "concrete", "stone", "velvet", "ceramic", "gold", "silver", "brass",
    "living room", "bedroom", "kitchen", "bathroom", "office", "outdoor",
    "garden", "restaurant", "cafe", "store",
    "sofa", "chair", "table", "bed", "lamp", "cabinet", "plant", "rug",
    "door", "window", "armchair", "bookshelf", "decor", "lighting", "desk",
    "fireplace", "appliance", "electronics", "kitchenware", "toy", "vehicle",
]

OBJECT_TYPE_TERMS = {
    "chair": ["chair", "armchair", "dining chair", "lounge chair", "bar chair", "stool", "bench"],
    "sofa": ["sofa", "couch", "sectional", "loveseat", "corner sofa"],
    "table": ["table", "coffee table", "dining table", "side table", "console", "desk"],
    "bed": ["bed", "headboard", "bunk bed", "nightstand", "bedside"],
    "cabinet": ["cabinet", "wardrobe", "closet", "dresser", "shelf", "bookshelf", "sideboard"],
    "lighting": ["lamp", "lighting", "chandelier", "pendant", "sconce", "spotlight"],
    "decor": ["decor", "vase", "mirror", "painting", "frame", "sculpture", "accessory"],
    "plant": ["plant", "tree", "flower", "potted"],
    "rug": ["rug", "carpet", "mat"],
    "kitchenware": ["kitchenware", "appliance", "sink", "faucet", "cooktop", "oven"],
    "door_window": ["door", "window", "partition", "shutter"],
    "vehicle": ["vehicle", "car", "bike", "motorcycle"],
}

STYLE_TERMS = {
    "modern": ["modern", "contemporary"],
    "classic": ["classic", "traditional", "baroque", "neoclassical"],
    "minimalist": ["minimalist", "minimal", "clean"],
    "scandinavian": ["scandinavian", "scandi", "nordic"],
    "industrial": ["industrial", "loft"],
    "luxury": ["luxury", "premium", "glam", "glamour", "gold"],
    "vintage": ["vintage", "retro", "mid century", "mid-century"],
    "rustic": ["rustic", "farmhouse", "country"],
    "bohemian": ["bohemian", "boho"],
    "futuristic": ["futuristic", "sci fi", "sci-fi"],
}

MATERIAL_TERMS = {
    "wood": ["wood", "wooden", "oak", "walnut", "ash", "teak"],
    "metal": ["metal", "steel", "iron", "aluminum", "chrome"],
    "glass": ["glass", "transparent"],
    "marble": ["marble", "travertine", "onyx"],
    "leather": ["leather", "suede"],
    "fabric": ["fabric", "linen", "cotton", "boucle", "textile"],
    "velvet": ["velvet"],
    "stone": ["stone", "concrete", "cement"],
    "ceramic": ["ceramic", "porcelain"],
    "brass": ["brass", "bronze", "copper"],
    "plastic": ["plastic", "polycarbonate", "acrylic"],
}

ROOM_TERMS = {
    "living_room": ["living room", "lounge", "salon"],
    "bedroom": ["bedroom", "kids room", "children room"],
    "kitchen": ["kitchen", "dining"],
    "bathroom": ["bathroom", "wc", "toilet"],
    "office": ["office", "workspace", "work space", "study"],
    "outdoor": ["outdoor", "garden", "terrace", "balcony", "patio"],
    "retail": ["store", "shop", "restaurant", "cafe", "hotel", "lobby"],
}

COLOR_TERMS = {
    "black": ["black", "dark"],
    "white": ["white", "ivory"],
    "gray": ["gray", "grey", "silver"],
    "beige": ["beige", "cream", "sand", "taupe"],
    "brown": ["brown", "walnut", "oak", "wood"],
    "green": ["green", "olive", "sage"],
    "blue": ["blue", "navy"],
    "red": ["red", "burgundy"],
    "yellow": ["yellow", "mustard"],
    "gold": ["gold", "brass"],
}

RENDER_TYPE_TERMS = {
    "scene_render": ["scene", "interior", "exterior", "room", "render"],
    "product_render": ["product", "pack", "set", "collection", "model"],
    "closeup": ["closeup", "close up", "detail"],
    "technical": ["cad", "obj", "fbx", "skp", "blend", "3ds", "max"],
}

DESIGN_SIGNAL_TERMS = set(DESIGN_TAXONOMY) | {
    "3d", "3ds", "3dsmax", "max", "model", "models", "scene", "scenes",
    "interior", "exterior", "architecture", "architectural", "decorative",
    "furniture", "fixture", "asset", "obj", "fbx", "skp", "blend",
    "corona", "vray", "v-ray", "render", "collection", "set",
}
DESIGN_SIGNAL_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(term) for term in sorted(DESIGN_SIGNAL_TERMS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

STATE_VERSION = 2
RESUME_CONFIRM_HITS = 25


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json_file(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else default
    except Exception:
        return default


def latest_log_file(data_path: Path) -> Optional[Path]:
    temp_dir = data_path / "temp"
    logs = list(temp_dir.glob("mega_sync*.err.log")) if temp_dir.exists() else []
    if not logs:
        return None
    return max(logs, key=lambda p: p.stat().st_mtime)


def build_status(data_path: Optional[Path] = None) -> Dict[str, Any]:
    data_path = data_path or (PROJECT_ROOT / "data")
    state = read_json_file(data_path / "sync_state.json", {})
    failed = read_json_file(data_path / "mega_failed.json", {"items": {}})
    channels = state.get("channels", {}) if isinstance(state.get("channels"), dict) else {}
    audit_channels = [data for data in channels.values() if isinstance(data, dict) and ("audit_cursor_id" in data or data.get("audit_done"))]
    failed_items = failed.get("items", {}) if isinstance(failed.get("items"), dict) else {}

    log_path = latest_log_file(data_path)
    log_lines: List[str] = []
    if log_path and log_path.exists():
        try:
            log_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            log_lines = []

    latest_channels = sorted(
        [
            {
                "channel_id": channel_id,
                "audit_cursor_id": data.get("audit_cursor_id"),
                "audit_done": bool(data.get("audit_done")),
                "audit_scanned": int(data.get("audit_scanned") or 0),
                "audit_updated_at": data.get("audit_updated_at"),
            }
            for channel_id, data in channels.items()
            if isinstance(data, dict) and ("audit_cursor_id" in data or data.get("audit_done"))
        ],
        key=lambda item: str(item.get("audit_updated_at")),
        reverse=True,
    )

    return {
        "generated_at": utc_now(),
        "audit_channels": len(audit_channels),
        "audit_done_channels": sum(1 for data in audit_channels if data.get("audit_done")),
        "audit_scanned_total": sum(int(data.get("audit_scanned") or 0) for data in audit_channels),
        "failed_count": len(failed_items),
        "failed_by_error": summarize_failed_errors(failed_items),
        "latest_channels": latest_channels[:15],
        "latest_log": str(log_path) if log_path else "",
        "latest_log_updated_at": log_path.stat().st_mtime if log_path and log_path.exists() else None,
        "log_processed": sum(1 for line in log_lines if "Islendi:" in line),
        "log_failed": sum(1 for line in log_lines if "MegaSync item basarisiz" in line),
        "log_audit_done": sum(1 for line in log_lines if "audit tamamlandi" in line),
        "log_flood_wait": sum(1 for line in log_lines if "flood wait" in line.lower()),
        "last_log_line": log_lines[-1] if log_lines else "",
    }


def summarize_failed_errors(items: Dict[str, Any]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for item in items.values():
        if not isinstance(item, dict):
            key = "unknown"
        else:
            error = str(item.get("error") or "unknown")
            key = error[:120]
        summary[key] = summary.get(key, 0) + 1
    return summary


def write_audit_report(data_path: Optional[Path] = None) -> Path:
    data_path = data_path or (PROJECT_ROOT / "data")
    report = build_status(data_path)
    report_path = data_path / "audit_report.json"
    temp_path = report_path.with_suffix(".json.tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    temp_path.replace(report_path)
    return report_path


def print_status(write_report: bool = False) -> None:
    data_path = PROJECT_ROOT / "data"
    status = build_status(data_path)
    print("MegaSync Durumu")
    print(f"  Audit kanali       : {status['audit_channels']}")
    print(f"  Biten kanal        : {status['audit_done_channels']}")
    print(f"  Taranan foto       : {status['audit_scanned_total']}")
    print(f"  Failed kayit       : {status['failed_count']}")
    print(f"  Log processed      : {status['log_processed']}")
    print(f"  Log failed         : {status['log_failed']}")
    print(f"  Log audit done     : {status['log_audit_done']}")
    print(f"  Log flood wait     : {status['log_flood_wait']}")
    print(f"  Son log            : {status['latest_log']}")
    print(f"  Son satir          : {status['last_log_line']}")
    print("  Son audit kanallari:")
    for item in status["latest_channels"][:8]:
        print(
            f"    {item['channel_id']} cursor={item['audit_cursor_id']} "
            f"done={item['audit_done']} scanned={item['audit_scanned']} updated={item['audit_updated_at']}"
        )
    if write_report:
        report_path = write_audit_report(data_path)
        print(f"  Rapor yazildi      : {report_path}")


class MegaSync:
    def __init__(
        self,
        *,
        dry_run: bool = False,
        limit: Optional[int] = None,
        global_limit: Optional[int] = None,
        backfill: bool = False,
        backfill_scan_limit: int = 500,
        audit_once: bool = False,
        audit_scan_limit: int = 50000,
        failed_retry_hours: float = 6.0,
        workers: int = 4,
        encode_workers: Optional[int] = None,
        retry: int = 3,
        scan_batch_size: int = 50,
        queue_size: Optional[int] = None,
        audit_resume_overlap: int = 10000,
    ):
        self.settings = get_settings()
        self.dry_run = dry_run
        self.limit = limit
        self.global_limit = global_limit
        self.backfill = backfill
        self.backfill_scan_limit = max(1, backfill_scan_limit)
        self.audit_once = audit_once
        self.audit_scan_limit = max(1, audit_scan_limit)
        self.failed_retry_hours = max(0.0, failed_retry_hours)
        self.max_workers = max(1, workers)
        self.max_encode_workers = max(1, encode_workers or min(2, self.max_workers))
        self.retry = max(1, retry)
        self.scan_batch_size = max(10, scan_batch_size)
        self.audit_resume_overlap = max(0, audit_resume_overlap)
        self.queue_size = max(100, queue_size or self.max_workers * 20)
        self.queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=self.queue_size)
        self.encode_semaphore = asyncio.Semaphore(self.max_encode_workers)
        self.state_lock = asyncio.Lock()
        self.failed_lock = asyncio.Lock()
        self.global_queued = 0
        self.stats = {
            "complete": 0,
            "queued": 0,
            "processed": 0,
            "failed": 0,
            "dry_run_candidates": 0,
            "repair_file": 0,
            "repair_db": 0,
            "audit_scanned": 0,
            "audit_done_channels": 0,
            "failed_skipped": 0,
            "quarantined": 0,
        }

        data_path = self.settings.get_data_path()
        self.state_path = data_path / "sync_state.json"
        self.failed_path = data_path / "mega_failed.json"
        self.report_path = data_path / "audit_report.json"
        self.temp_download_dir = data_path / "temp" / "mega_downloads"
        self.corrupt_dir = data_path / "corrupt_images"

        self.store = ChromaStore(self.settings.chroma_db_path, self.settings.get_collection_name())
        self.manager = TelegramClientManager(
            api_id=self.settings.telegram_api_id,
            api_hash=self.settings.telegram_api_hash,
            phone=self.settings.telegram_phone,
            session_dir=str(data_path),
        )

        self.state = self.load_state()
        self.failed = self.load_failed()
        self.taxonomy_matrix = None
        self.taxonomy_names: List[str] = []

        if not self.dry_run:
            self.temp_download_dir.mkdir(parents=True, exist_ok=True)
            self.corrupt_dir.mkdir(parents=True, exist_ok=True)
            self.load_taxonomy_model()

    def load_taxonomy_model(self) -> None:
        load_model(self.settings.clip_model_name, self.settings.clip_pretrained)

        import numpy as np

        taxonomy_vectors = []
        for term in DESIGN_TAXONOMY:
            vec = encode_text(
                f"a photo of a {term}",
                model_name=self.settings.clip_model_name,
                pretrained=self.settings.clip_pretrained,
            )
            if vec is not None:
                taxonomy_vectors.append(vec)
                self.taxonomy_names.append(term)

        if taxonomy_vectors:
            self.taxonomy_matrix = np.vstack(taxonomy_vectors)

    def load_json_file(self, path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
        if not path.exists():
            return default
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else default
        except Exception as exc:
            logger.warning("JSON okunamadi (%s): %s", path, exc)
            return default

    def atomic_write_json(self, path: Path, data: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        temp_path.replace(path)

    def load_state(self) -> Dict[str, Any]:
        raw = self.load_json_file(self.state_path, {})
        if raw.get("version") == STATE_VERSION and isinstance(raw.get("channels"), dict):
            return raw

        channels = {}
        for channel_id, last_id in raw.items():
            if isinstance(last_id, int):
                channels[str(channel_id)] = {
                    "last_complete_id": last_id,
                    "legacy_boundary": True,
                    "updated_at": None,
                }

        return {
            "version": STATE_VERSION,
            "channels": channels,
            "migrated_from_legacy": bool(channels),
        }

    def load_failed(self) -> Dict[str, Any]:
        data = self.load_json_file(self.failed_path, {})
        if data.get("version") == STATE_VERSION and isinstance(data.get("items"), dict):
            return data
        if all(isinstance(v, dict) for v in data.values()):
            return {"version": STATE_VERSION, "items": data}
        return {"version": STATE_VERSION, "items": {}}

    async def save_state(self) -> None:
        if not self.dry_run:
            async with self.state_lock:
                self.atomic_write_json(self.state_path, self.state)

    async def save_failed(self) -> None:
        if not self.dry_run:
            async with self.failed_lock:
                self.atomic_write_json(self.failed_path, self.failed)

    def channel_state(self, channel_id: str) -> Dict[str, Any]:
        channels = self.state.setdefault("channels", {})
        return channels.setdefault(str(channel_id), {"last_complete_id": 0, "updated_at": None})

    async def mark_complete(
        self,
        channel_id: str,
        message_id: int,
        doc_id: str,
        *,
        count_processed: bool = True,
    ) -> None:
        if count_processed:
            self.stats["processed"] += 1
        if doc_id in self.failed.get("items", {}):
            self.failed["items"].pop(doc_id, None)
            await self.save_failed()

        state = self.channel_state(channel_id)
        current = int(state.get("last_complete_id") or 0)
        if message_id > current:
            state["last_complete_id"] = message_id
        state["legacy_boundary"] = False
        state["updated_at"] = utc_now()
        await self.save_state()

    async def record_failed(
        self,
        item: Dict[str, Any],
        *,
        stage: str,
        error: str,
        attempts: int,
    ) -> None:
        self.stats["failed"] += 1
        if self.dry_run:
            return

        failed_items = self.failed.setdefault("items", {})
        previous = failed_items.get(item["doc_id"], {})
        failed_items[item["doc_id"]] = {
            "doc_id": item["doc_id"],
            "channel_id": item["numeric_id"],
            "message_id": item["message"].id,
            "link": item["link"],
            "stage": stage,
            "error": error,
            "attempts": int(previous.get("attempts") or 0) + attempts,
            "last_seen_at": utc_now(),
        }
        await self.save_failed()

    def is_valid_image(self, path: Path) -> bool:
        if not path.exists() or path.stat().st_size <= 0:
            return False
        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                img.load()
                if img.width < 16 or img.height < 16:
                    return False
                img.convert("RGB")
            return True
        except Exception:
            return False

    def quarantine_image(self, path: Path, doc_id: str, reason: str) -> Optional[Path]:
        if self.dry_run or not path.exists():
            return None
        self.corrupt_dir.mkdir(parents=True, exist_ok=True)
        safe_reason = re.sub(r"[^a-zA-Z0-9_-]+", "_", reason).strip("_") or "corrupt"
        target = self.corrupt_dir / f"{doc_id}_{int(time.time())}_{safe_reason}{path.suffix or '.jpg'}"
        try:
            path.replace(target)
            self.stats["quarantined"] += 1
            logger.warning("Bozuk gorsel karantinaya alindi: %s -> %s", path, target)
            return target
        except Exception as exc:
            logger.warning("Bozuk gorsel karantinaya alinamadi (%s): %s", path, exc)
            return None

    def ids_exist(self, doc_ids: Iterable[str]) -> Set[str]:
        ids = list(dict.fromkeys(doc_ids))
        if not ids:
            return set()
        try:
            result = self.store.collection.get(ids=ids, include=[])
            return set(result.get("ids", [])) if result else set()
        except Exception as exc:
            logger.warning("Chroma batch get hata verdi, tekil fallback deneniyor: %s", exc)

        existing: Set[str] = set()
        for doc_id in ids:
            try:
                result = self.store.collection.get(ids=[doc_id], include=[])
                if result and result.get("ids"):
                    existing.add(doc_id)
            except Exception as exc:
                logger.warning("Chroma id okunamadi (%s): %s", doc_id, exc)
        return existing

    def inspect_record(self, doc_id: str, save_path: Path, existing_ids: Optional[Set[str]] = None) -> Dict[str, bool]:
        file_ok = self.is_valid_image(save_path)
        db_ok = doc_id in existing_ids if existing_ids is not None else doc_id in self.ids_exist([doc_id])
        return {"file_ok": file_ok, "db_ok": db_ok, "complete": file_ok and db_ok}

    def make_link(self, username: str, numeric_id: str, message_id: int) -> str:
        if username:
            return f"https://t.me/{username}/{message_id}"
        return f"https://t.me/c/{numeric_id}/{message_id}"

    def global_capacity_reached(self) -> bool:
        return self.global_limit is not None and self.global_queued >= self.global_limit

    def parse_timestamp(self, value: Any) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None

    def should_retry_failed(self, failed: Dict[str, Any]) -> bool:
        if self.failed_retry_hours <= 0:
            return True
        last_seen = self.parse_timestamp(failed.get("last_seen_at"))
        if last_seen is None:
            return True
        return datetime.now(timezone.utc) - last_seen >= timedelta(hours=self.failed_retry_hours)

    def get_backfill_offset_id(self, state: Dict[str, Any]) -> int:
        if not self.backfill:
            return 0
        cursor = int(state.get("backfill_cursor_id") or 0)
        return cursor if cursor > 0 else int(state.get("last_complete_id") or 0)

    def get_audit_offset_id(self, state: Dict[str, Any]) -> int:
        if not self.audit_once:
            return 0
        cursor = int(state.get("audit_cursor_id") or 0)
        if cursor <= 0:
            return 0
        # Re-scan a small message-id window after restarts so queued-but-not-finished
        # items cannot be skipped if the process is stopped mid-channel.
        return cursor + self.audit_resume_overlap

    async def update_backfill_cursor(self, state: Dict[str, Any], message_id: int) -> None:
        if not self.backfill or self.dry_run:
            return
        state["backfill_cursor_id"] = message_id
        state["backfill_updated_at"] = utc_now()
        await self.save_state()

    async def update_audit_progress(
        self,
        state: Dict[str, Any],
        *,
        cursor_id: Optional[int] = None,
        scanned_delta: int = 0,
        done: bool = False,
    ) -> None:
        if not self.audit_once or self.dry_run:
            return
        if cursor_id is not None:
            state["audit_cursor_id"] = cursor_id
        if scanned_delta:
            state["audit_scanned"] = int(state.get("audit_scanned") or 0) + scanned_delta
        state["audit_done"] = bool(done)
        state["audit_updated_at"] = utc_now()
        await self.save_state()

    def build_item(self, message: Any, numeric_id: str, title: str, username: str) -> Dict[str, Any]:
        doc_id = f"{numeric_id}_{message.id}"
        return {
            "message": message,
            "numeric_id": numeric_id,
            "title": title,
            "username": username,
            "link": self.make_link(username, numeric_id, message.id),
            "doc_id": doc_id,
            "save_path": self.settings.get_images_path() / f"{doc_id}.jpg",
        }

    def has_design_signal(self, text: str) -> bool:
        return bool(text and DESIGN_SIGNAL_PATTERN.search(text))

    def normalize_design_text(self, text: str) -> str:
        normalized = re.sub(r"[_\-]+", " ", (text or "").lower())
        normalized = re.sub(r"[^a-z0-9 ]+", " ", normalized)
        return f" {re.sub(r'\s+', ' ', normalized).strip()} "

    def match_design_term(self, text: str, mapping: Dict[str, List[str]]) -> str:
        haystack = self.normalize_design_text(text)
        if not haystack.strip():
            return ""
        for label, terms in mapping.items():
            for term in terms:
                needle = self.normalize_design_text(term)
                if needle.strip() and needle in haystack:
                    return label
        return ""

    def infer_design_metadata(
        self,
        original_text: str,
        ai_description: str,
        caption_has_signal: bool,
    ) -> Dict[str, Any]:
        caption_text = original_text if caption_has_signal else ""
        combined_text = f"{original_text} {ai_description}"
        source_parts: List[str] = []
        if caption_has_signal:
            source_parts.append("caption")
        if ai_description:
            source_parts.append("vision_taxonomy")

        object_type = self.match_design_term(caption_text, OBJECT_TYPE_TERMS) or self.match_design_term(combined_text, OBJECT_TYPE_TERMS)
        style = self.match_design_term(caption_text, STYLE_TERMS) or self.match_design_term(combined_text, STYLE_TERMS)
        material = self.match_design_term(caption_text, MATERIAL_TERMS) or self.match_design_term(combined_text, MATERIAL_TERMS)
        room = self.match_design_term(caption_text, ROOM_TERMS) or self.match_design_term(combined_text, ROOM_TERMS)
        color = self.match_design_term(caption_text, COLOR_TERMS) or self.match_design_term(combined_text, COLOR_TERMS)
        render_type = self.match_design_term(caption_text, RENDER_TYPE_TERMS) or self.match_design_term(combined_text, RENDER_TYPE_TERMS)

        populated = [object_type, style, material, room, color, render_type]
        confidence = 0.0
        if any(populated):
            confidence = 0.92 if caption_has_signal else 0.68

        quality_tags = []
        if caption_has_signal:
            quality_tags.append("caption_signal")
        if ai_description:
            quality_tags.append("vision_taxonomy")
        if any(populated):
            quality_tags.append("rich_metadata")

        return {
            MetadataSchema.OBJECT_TYPE: object_type,
            MetadataSchema.STYLE: style,
            MetadataSchema.MATERIAL: material,
            MetadataSchema.ROOM: room,
            MetadataSchema.COLOR_FAMILY: color,
            MetadataSchema.RENDER_TYPE: render_type,
            MetadataSchema.CATEGORY_SOURCE: "+".join(source_parts) if source_parts else "unknown",
            MetadataSchema.CATEGORY_CONFIDENCE: confidence,
            MetadataSchema.QUALITY_TAGS: ", ".join(quality_tags),
        }

    async def tag_image(self, img_vec: Any) -> List[str]:
        if self.taxonomy_matrix is None:
            return []

        import numpy as np

        similarities = np.dot(self.taxonomy_matrix, img_vec)
        top_indices = np.where(similarities > 0.21)[0]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        return [self.taxonomy_names[i] for i in top_indices[:5]]

    async def download_atomic(self, message: Any, doc_id: str, save_path: Path) -> None:
        if self.is_valid_image(save_path):
            return
        if save_path.exists():
            self.quarantine_image(save_path, doc_id, "invalid_existing_file")

        temp_path = self.temp_download_dir / f"{doc_id}.part"
        if temp_path.exists():
            temp_path.unlink()

        save_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            await message.download_media(file=str(temp_path))

            if not self.is_valid_image(temp_path):
                raise RuntimeError("indirilen gecici dosya gecerli bir gorsel degil")

            temp_path.replace(save_path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

    async def process_item(self, item: Dict[str, Any]) -> None:
        message = item["message"]
        doc_id = item["doc_id"]
        save_path = item["save_path"]

        await self.download_atomic(message, doc_id, save_path)

        async with self.encode_semaphore:
            img_vec = await asyncio.to_thread(
                encode_image,
                str(save_path),
                model_name=self.settings.clip_model_name,
                pretrained=self.settings.clip_pretrained,
            )
        if img_vec is None:
            self.quarantine_image(save_path, doc_id, "clip_encode_failed")
            raise RuntimeError("CLIP encode sonucu bos dondu")

        original_text = message.text if message.text else ""
        ai_description = ""
        caption_has_signal = self.has_design_signal(original_text)
        if not caption_has_signal:
            ai_tags = await self.tag_image(img_vec)
            ai_description = ", ".join(ai_tags)

        design_metadata = self.infer_design_metadata(original_text, ai_description, caption_has_signal)
        design_document = " ".join(str(value) for value in design_metadata.values() if value)

        metadata = {
            MetadataSchema.CHANNEL_ID: item["numeric_id"],
            MetadataSchema.MESSAGE_ID: message.id,
            MetadataSchema.IMAGE_PATH: str(save_path),
            MetadataSchema.DEEP_LINK: item["link"],
            MetadataSchema.CHANNEL_TITLE: item["title"],
            MetadataSchema.CHANNEL_USERNAME: item["username"],
            MetadataSchema.TIMESTAMP: message.date.isoformat(),
            MetadataSchema.DEEP_TAGS: ai_description,
            MetadataSchema.CAPTION: original_text[:500],
            MetadataSchema.SOURCE: "MegaSync",
            MetadataSchema.MODEL_NAME: self.settings.clip_model_name,
            MetadataSchema.SCHEMA_VERSION_KEY: MetadataSchema.CURRENT_SCHEMA_VERSION,
            **design_metadata,
        }

        added = self.store.add(
            doc_id=doc_id,
            embedding=img_vec,
            metadata=metadata,
            document=f"{original_text}\nAI Tags: {ai_description}\nDesign Metadata: {design_document}",
        )
        if not added or doc_id not in self.ids_exist([doc_id]):
            raise RuntimeError("Chroma upsert dogrulanamadi")

        await self.mark_complete(item["numeric_id"], message.id, doc_id)
        logger.info("Islendi: %s (Text: %s, AI: %s, caption_signal=%s)", doc_id, len(original_text), len(ai_description), caption_has_signal)

    async def worker(self) -> None:
        while True:
            item = await self.queue.get()
            try:
                last_error = ""
                last_stage = "process"
                success = False
                for attempt in range(1, self.retry + 1):
                    try:
                        await self.process_item(item)
                        success = True
                        break
                    except Exception as exc:
                        last_error = str(exc)
                        last_stage = "process"
                        if attempt < self.retry:
                            await asyncio.sleep(min(2 ** attempt, 10))

                if success:
                    continue

                await self.record_failed(
                    item,
                    stage=last_stage,
                    error=last_error,
                    attempts=self.retry,
                )
                logger.error("MegaSync item basarisiz: %s - %s", item["doc_id"], last_error)
            finally:
                self.queue.task_done()

    async def enqueue_item(self, item: Dict[str, Any], status: Dict[str, bool], queued_ids: Set[str]) -> bool:
        if item["doc_id"] in queued_ids:
            return False
        if status["complete"]:
            self.stats["complete"] += 1
            if not self.audit_once:
                await self.mark_complete(
                    item["numeric_id"],
                    item["message"].id,
                    item["doc_id"],
                    count_processed=False,
                )
            return False

        if self.global_capacity_reached():
            return False

        if status["db_ok"] and not status["file_ok"]:
            self.stats["repair_file"] += 1
        elif status["file_ok"] and not status["db_ok"]:
            self.stats["repair_db"] += 1

        if self.dry_run:
            self.stats["dry_run_candidates"] += 1
            logger.info(
                "DRY-RUN aday: %s file_ok=%s db_ok=%s",
                item["doc_id"],
                status["file_ok"],
                status["db_ok"],
            )
            self.global_queued += 1
            return True

        queued_ids.add(item["doc_id"])
        await self.queue.put(item)
        self.stats["queued"] += 1
        self.global_queued += 1
        return True

    async def enqueue_failed_first(
        self,
        client: Any,
        entity: Any,
        numeric_id: str,
        title: str,
        username: str,
        queued_ids: Set[str],
        remaining: Optional[int],
    ) -> int:
        failed_items = [
            item
            for item in self.failed.get("items", {}).values()
            if str(item.get("channel_id")) == str(numeric_id)
        ]
        queued_count = 0
        for failed in failed_items:
            if self.global_capacity_reached():
                break
            if remaining is not None and queued_count >= remaining:
                break
            if not self.should_retry_failed(failed):
                self.stats["failed_skipped"] += 1
                continue
            message_id = failed.get("message_id")
            if not isinstance(message_id, int):
                continue
            message = await client.get_messages(entity, ids=message_id)
            if not message or not getattr(message, "photo", None):
                continue
            item = self.build_item(message, numeric_id, title, username)
            status = self.inspect_record(item["doc_id"], item["save_path"])
            if await self.enqueue_item(item, status, queued_ids):
                queued_count += 1
        return queued_count

    async def scan_channel(self, client: Any, channel_id: str) -> None:
        logger.info("Kanal taraniyor: %s", channel_id)
        entity = await client.get_entity(channel_id)
        numeric_id = str(entity.id).replace("-100", "").replace("-", "")
        title = getattr(entity, "title", "Bilinmeyen")
        username = getattr(entity, "username", "")
        state = self.channel_state(numeric_id)

        if self.audit_once and state.get("audit_done"):
            self.stats["audit_done_channels"] += 1
            logger.info("%s audit zaten tamamlanmis, atlandi.", title)
            return

        resume_boundary = int(state.get("last_complete_id") or 0)
        if self.audit_once:
            offset_id = self.get_audit_offset_id(state)
            scan_limit = self.audit_scan_limit
        elif self.backfill:
            offset_id = self.get_backfill_offset_id(state)
            scan_limit = self.backfill_scan_limit
        else:
            offset_id = 0
            scan_limit = None

        queued_ids: Set[str] = set()
        queued_for_channel = await self.enqueue_failed_first(
            client,
            entity,
            numeric_id,
            title,
            username,
            queued_ids,
            self.limit,
        )
        resume_hits = 0
        batch: List[Any] = []
        inspected_for_channel = 0
        stopped_early = False
        last_cursor_id: Optional[int] = None

        async def save_scan_cursor(cursor_id: int, scanned_delta: int) -> None:
            if self.audit_once:
                self.stats["audit_scanned"] += scanned_delta
                await self.update_audit_progress(
                    state,
                    cursor_id=cursor_id,
                    scanned_delta=scanned_delta,
                    done=False,
                )
            elif self.backfill:
                await self.update_backfill_cursor(state, cursor_id)

        async def flush_batch(messages: List[Any]) -> tuple[bool, bool]:
            nonlocal queued_for_channel, resume_hits
            if not messages:
                return False, True

            items = [self.build_item(message, numeric_id, title, username) for message in messages]
            existing_ids = self.ids_exist([item["doc_id"] for item in items])

            for item in items:
                if self.global_capacity_reached():
                    return True, False
                if self.limit is not None and queued_for_channel >= self.limit:
                    return True, False

                status = self.inspect_record(item["doc_id"], item["save_path"], existing_ids)
                if (
                    not self.backfill
                    and not self.audit_once
                    and item["message"].id <= resume_boundary
                    and status["complete"]
                ):
                    resume_hits += 1
                    if resume_hits >= RESUME_CONFIRM_HITS:
                        logger.info("%s resume sinirinda %s dogrulanmis kayit sonrasi durdu.", title, resume_hits)
                        return True, True
                elif not status["complete"]:
                    resume_hits = 0

                queued = await self.enqueue_item(item, status, queued_ids)
                if queued:
                    queued_for_channel += 1

            return False, True

        async for message in client.iter_messages(entity, offset_id=offset_id):
            if not getattr(message, "photo", None):
                continue

            batch.append(message)
            inspected_for_channel += 1
            last_cursor_id = message.id

            if len(batch) >= self.scan_batch_size:
                should_stop, fully_scanned = await flush_batch(batch)
                if fully_scanned:
                    await save_scan_cursor(batch[-1].id, len(batch))
                batch = []
                if should_stop:
                    stopped_early = True
                    break

            if scan_limit is not None and inspected_for_channel >= scan_limit:
                if batch:
                    should_stop, fully_scanned = await flush_batch(batch)
                    if fully_scanned:
                        await save_scan_cursor(batch[-1].id, len(batch))
                    batch = []
                    if should_stop:
                        stopped_early = True
                        break
                logger.info(
                    "%s %s scan limit %s doldu. Cursor=%s",
                    title,
                    "audit" if self.audit_once else "backfill",
                    scan_limit,
                    last_cursor_id,
                )
                stopped_early = True
                break

            if self.dry_run and self.limit is not None and inspected_for_channel >= self.limit:
                if batch:
                    await flush_batch(batch)
                    batch = []
                stopped_early = True
                break

        if batch:
            should_stop, fully_scanned = await flush_batch(batch)
            if fully_scanned:
                await save_scan_cursor(batch[-1].id, len(batch))
            if should_stop:
                stopped_early = True

        if self.audit_once and not stopped_early:
            await self.update_audit_progress(
                state,
                cursor_id=last_cursor_id,
                scanned_delta=0,
                done=True,
            )
            self.stats["audit_done_channels"] += 1
            logger.info("%s audit tamamlandi. Toplam audit_scanned=%s", title, state.get("audit_scanned", 0))

        logger.info(
            "%s tarama bitti. Kuyruga alinan/adayi: %s, incelenen_foto: %s",
            title,
            queued_for_channel,
            inspected_for_channel,
        )

    async def run(self) -> None:
        mode = "DRY-RUN" if self.dry_run else "GERCEK"
        started_at = time.perf_counter()
        logger.info(
            "MegaSync baslatildi. Mod=%s workers=%s encode_workers=%s retry=%s queue_size=%s scan_batch_size=%s audit_resume_overlap=%s limit=%s global_limit=%s backfill=%s backfill_scan_limit=%s audit_once=%s audit_scan_limit=%s failed_retry_hours=%s",
            mode,
            self.max_workers,
            self.max_encode_workers,
            self.retry,
            self.queue_size,
            self.scan_batch_size,
            self.audit_resume_overlap,
            self.limit,
            self.global_limit,
            self.backfill,
            self.backfill_scan_limit,
            self.audit_once,
            self.audit_scan_limit,
            self.failed_retry_hours,
        )

        async with self.manager as client:
            workers: List[asyncio.Task] = []
            if not self.dry_run:
                workers = [asyncio.create_task(self.worker()) for _ in range(self.max_workers)]

            for channel_id in self.settings.get_channel_list():
                try:
                    await self.scan_channel(client, channel_id)
                    if self.global_capacity_reached():
                        logger.info("Global limit %s doldu, yeni kanal taramasi durduruluyor.", self.global_limit)
                        break
                except Exception as exc:
                    logger.error("%s kanal hatasi: %s", channel_id, exc)
                    await asyncio.sleep(5)

            if not self.dry_run:
                await self.queue.join()
                for task in workers:
                    task.cancel()
                await asyncio.gather(*workers, return_exceptions=True)

        elapsed = max(time.perf_counter() - started_at, 0.001)
        rate = (self.stats["processed"] / elapsed) * 60
        logger.info("MegaSync tamamlandi: %s | sure=%.1fs hiz=%.1f yeni_kayit/dk", self.stats, elapsed, rate)
        if self.audit_once and not self.dry_run:
            report_path = write_audit_report(self.settings.get_data_path())
            logger.info("Audit raporu yazildi: %s", report_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dayanikli Telegram MegaSync - parametresiz calistirinca tam audit yapar")
    parser.add_argument("--dry-run", action="store_true", help="Indirme/DB/state yazmadan adaylari raporla")
    parser.add_argument("--limit", type=int, default=None, help="Kanal basina kuyruga alinacak aday sayisi")
    parser.add_argument("--global-limit", type=int, default=None, help="Tum calisma icin toplam aday sayisi")
    parser.add_argument("--backfill", action="store_true", help="Resume sinirinda durmadan gecmisteki eksik kayitlari ara")
    parser.add_argument("--backfill-scan-limit", type=int, default=500, help="Backfill modunda kanal basina incelenecek foto sayisi")
    parser.add_argument("--audit-once", action="store_true", default=True, help="Tum gecmis arsivi bir kere cursor ile denetle (varsayilan acik)")
    parser.add_argument("--audit-scan-limit", type=int, default=999999, help="Audit modunda kanal basina incelenecek foto sayisi")
    parser.add_argument("--failed-retry-hours", type=float, default=6.0, help="Failed kayitlari tekrar denemeden once beklenecek saat")
    parser.add_argument("--speed", choices=["safe", "fast", "aggressive"], default="fast", help="Worker/batch preset. Parametresiz calistirma fast preset kullanir")
    parser.add_argument("--workers", type=int, default=None, help="Paralel download/process worker sayisi")
    parser.add_argument("--encode-workers", type=int, default=None, help="Ayni anda calisacak CLIP encode sayisi")
    parser.add_argument("--retry", type=int, default=None, help="Basarisiz isler icin deneme sayisi")
    parser.add_argument("--scan-batch-size", type=int, default=None, help="Chroma/file kontrol batch boyutu")
    parser.add_argument("--queue-size", type=int, default=None, help="Download/encode kuyruk kapasitesi")
    parser.add_argument("--audit-resume-overlap", type=int, default=10000, help="Restart sonrasi audit cursor'unu bu kadar message id geri sar")
    parser.add_argument("--normal-sync", action="store_true", help="Audit yerine sadece yeni mesajlar icin normal sync calistir")
    parser.add_argument("--status", action="store_true", help="MegaSync audit/state durumunu yazdir ve cik")
    parser.add_argument("--write-report", action="store_true", help="Audit durum raporunu data/audit_report.json olarak yaz")
    args = parser.parse_args()
    presets = {
        "safe": {"workers": 12, "encode_workers": 2, "retry": 3, "scan_batch_size": 50, "queue_size": 300},
        "fast": {"workers": 24, "encode_workers": 4, "retry": 3, "scan_batch_size": 150, "queue_size": 1000},
        "aggressive": {"workers": 36, "encode_workers": 6, "retry": 2, "scan_batch_size": 250, "queue_size": 2000},
    }
    preset = presets[args.speed]
    if args.workers is None:
        args.workers = preset["workers"]
    if args.encode_workers is None:
        args.encode_workers = preset["encode_workers"]
    if args.retry is None:
        args.retry = preset["retry"]
    if args.scan_batch_size is None:
        args.scan_batch_size = preset["scan_batch_size"]
    if args.queue_size is None:
        args.queue_size = preset["queue_size"]
    if args.normal_sync:
        args.audit_once = False
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.status or args.write_report:
        print_status(write_report=args.write_report)
        raise SystemExit(0)

    sync = MegaSync(
        dry_run=args.dry_run,
        limit=args.limit,
        global_limit=args.global_limit,
        backfill=args.backfill,
        backfill_scan_limit=args.backfill_scan_limit,
        audit_once=args.audit_once,
        audit_scan_limit=args.audit_scan_limit,
        failed_retry_hours=args.failed_retry_hours,
        workers=args.workers,
        encode_workers=args.encode_workers,
        retry=args.retry,
        scan_batch_size=args.scan_batch_size,
        queue_size=args.queue_size,
        audit_resume_overlap=args.audit_resume_overlap,
    )
    asyncio.run(sync.run())
