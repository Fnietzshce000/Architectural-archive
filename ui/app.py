"""
Streamlit Ana Uygulama — 3D Model Arama Motoru UI.
Metin ve görsel tabanlı çok kipli arama arayüzü.
"""
import os
import sys
import json
import logging
import subprocess
from datetime import datetime
from collections import Counter
from pathlib import Path

import streamlit as st
from PIL import Image

# Proje kökünü Python path'e ekle
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_settings
from indexer.chroma_store import ChromaStore
from search.text_search import search_by_text
from search.image_search import search_by_image

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════
# Sayfa Yapılandırması
# ══════════════════════════════════════════════════

st.set_page_config(
    page_title="3D Model Arama Motoru",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════
# Özel CSS — Premium Koyu Tema
# ══════════════════════════════════════════════════

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ── Genel Tema ── */
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Inter', sans-serif !important;
    }
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    /* ── Başlık Stili ── */
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .subtitle {
        text-align: center;
        color: #8899aa;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* ── Arama Kutusu ── */
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 12px !important;
        color: #e0e0e0 !important;
        font-size: 1.1rem !important;
        padding: 12px 16px !important;
        transition: all 0.3s ease;
    }
    .stTextInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.2) !important;
    }

    /* ── Sonuç Kartları ── */
    .result-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 12px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 12px;
    }
    .result-card:hover {
        background: rgba(102, 126, 234, 0.08);
        border-color: rgba(102, 126, 234, 0.3);
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.15);
    }

    /* ── Skor ve Kanal Rozetleri ── */
    .badge-container {
        display: flex;
        gap: 6px;
        margin-top: 8px;
        flex-wrap: wrap;
    }
    .score-badge {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 600;
    }
    .aesthetic-badge {
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.65rem;
        font-weight: 600;
        color: white;
    }
    .aesthetic-elite { background: linear-gradient(135deg, #f5af19, #f12711); }
    .aesthetic-good { background: linear-gradient(135deg, #56ab2f, #a8e063); color: #1a1a2e; }
    .aesthetic-mid { background: rgba(255,255,255,0.1); color: #8899aa; }
    .aesthetic-low { background: rgba(255,50,50,0.15); color: #ff6b6b; }
    .channel-tag {
        background: rgba(255, 255, 255, 0.05);
        color: #8899aa;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.65rem;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .ai-tag {
        background: rgba(102, 126, 234, 0.12);
        color: #99aaff;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.6rem;
        font-weight: 500;
        border: 1px solid rgba(102, 126, 234, 0.15);
    }

    /* ── İstatistik Kartları ── */
    .stat-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #8899aa;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: rgba(10, 10, 15, 0.95) !important;
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    /* ── Link Butonları ── */
    .telegram-btn {
        display: inline-block;
        background: linear-gradient(135deg, #0088cc, #0099ff);
        color: white !important;
        padding: 6px 14px;
        border-radius: 10px;
        text-decoration: none;
        font-size: 0.85rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    .telegram-btn:hover {
        background: linear-gradient(135deg, #0099ff, #33bbff);
        box-shadow: 0 4px 15px rgba(0, 136, 204, 0.4);
    }

    /* ── Dosya Adı ── */
    .file-name {
        color: #ccddee;
        font-size: 0.8rem;
        margin-top: 4px;
        word-break: break-all;
    }

    /* ── Boş Durum ── */
    .empty-state {
        text-align: center;
        padding: 60px 20px;
        color: #556677;
    }
    .empty-state-emoji {
        font-size: 4rem;
        margin-bottom: 16px;
    }
    /* ── Sohbet Balonları ── */
    [data-testid="stChatMessage"] {
        border-radius: 18px !important;
        margin-bottom: 1.2rem !important;
        padding: 1.2rem 1.4rem !important;
        backdrop-filter: blur(10px) !important;
        animation: fadeInUp 0.3s ease-out;
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    /* Asistan Balonları */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.08), rgba(118, 75, 162, 0.05)) !important;
        border: 1px solid rgba(102, 126, 234, 0.18) !important;
        border-left: 3px solid #667eea !important;
    }
    /* Kullanıcı Balonları */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-right: 3px solid rgba(240, 147, 251, 0.5) !important;
    }
    [data-testid="stChatMessageContent"] p {
        font-size: 0.95rem !important;
        line-height: 1.7 !important;
        color: #ccddee !important;
    }
    /* Avatar Stil */
    [data-testid="stChatMessage"] [data-testid*="chatAvatar"] {
        width: 36px !important;
        height: 36px !important;
        font-size: 1.2rem !important;
    }

    /* ── Chat İçindeki Model Galerisi ── */
    .chat-result-card {
        background: rgba(10, 10, 15, 0.5);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 14px;
        overflow: hidden;
        margin-bottom: 10px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .chat-result-card:hover {
        transform: translateY(-3px) scale(1.02);
        border-color: #667eea;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
    }
    .chat-result-image {
        width: 100%;
        height: 130px;
        object-fit: cover;
    }
    .chat-result-btn {
        width: 100%;
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        padding: 6px !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        border-radius: 0 0 14px 14px !important;
        transition: opacity 0.2s !important;
    }
    .chat-result-btn:hover {
        opacity: 0.85 !important;
    }

    /* ── Chat Input ── */
    [data-testid="stChatInput"] {
        background-color: rgba(10, 10, 15, 0.95) !important;
        border-top: 1px solid rgba(102, 126, 234, 0.15) !important;
        backdrop-filter: blur(10px) !important;
    }
    [data-testid="stChatInput"] textarea {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(102, 126, 234, 0.2) !important;
        border-radius: 12px !important;
        color: #e0e0e0 !important;
        font-family: 'Inter', sans-serif !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 15px rgba(102, 126, 234, 0.15) !important;
    }

    /* ── Chat Model Galerisi Başlığı ── */
    .chat-gallery-header {
        margin-top: 12px;
        margin-bottom: 8px;
        font-size: 0.85rem;
        color: #667eea;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 6px 10px;
        background: rgba(102, 126, 234, 0.06);
        border-radius: 8px;
        border-left: 3px solid #667eea;
    }
    /* ── Sidebar Tasarımı ── */
    [data-testid="stSidebar"] {
        background-color: #0a0a0f !important;
        border-right: 1px solid rgba(102, 126, 234, 0.1) !important;
    }
    .side-metric-card {
        background: linear-gradient(145deg, rgba(20, 20, 30, 0.9), rgba(10, 10, 15, 0.9));
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        margin-bottom: 10px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    .side-metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #667eea;
        margin-bottom: 2px;
        text-shadow: 0 0 10px rgba(102, 126, 234, 0.3);
    }
    .side-metric-label {
        font-size: 0.75rem;
        color: #8899aa;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .sidebar-section-header {
        font-size: 0.9rem;
        font-weight: 700;
        color: #ccddee;
        margin: 20px 0 10px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .sidebar-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.3), transparent);
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════
# Session State Başlatma
# ══════════════════════════════════════════════════

if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "search_query" not in st.session_state:
    st.session_state.search_query = ""
if "search_mode" not in st.session_state:
    st.session_state.search_mode = 0  # 0: Metin, 1: Görsel
if "current_page" not in st.session_state:
    st.session_state.current_page = 0
if "last_query_key" not in st.session_state:
    st.session_state.last_query_key = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Merhaba dostum! 👋\n\nBen **Archi**, senin kişisel 3D model asistanıyım. 460 binden fazla modelin olduğu devasa arşivde arama yapabilirim.\n\n💡 **İpucu:** Bana doğal konuş — *\"Salon için modern bir koltuk bul\"* veya *\"Ahşap bacaklı minimalist masa lazım\"* gibi. Referans fotoğraf da yükleyebilirsin!", "results": []}
    ]


# ══════════════════════════════════════════════════
# ChromaDB Bağlantısı
# ══════════════════════════════════════════════════

@st.cache_resource
def get_store():
    """ChromaDB bağlantısını cache'ler."""
    settings = get_settings()
    return ChromaStore(
        db_path=settings.chroma_db_path,
        collection_name=settings.get_collection_name(),
    )


store = get_store()


@st.cache_data(ttl=45)
def read_json_cached(path_str: str, default: dict) -> dict:
    path = Path(path_str)
    if not path.exists():
        return default
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else default
    except Exception:
        return default


@st.cache_data(ttl=60)
def scan_archive_files(data_dir_str: str) -> dict:
    data_dir = Path(data_dir_str)
    images_dir = data_dir / "images"
    corrupt_dir = data_dir / "corrupt_images"
    temp_dir = data_dir / "temp" / "mega_downloads"

    image_count = 0
    image_bytes = 0
    if images_dir.exists():
        for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            for file_path in images_dir.glob(pattern):
                try:
                    image_count += 1
                    image_bytes += file_path.stat().st_size
                except OSError:
                    pass

    corrupt_count = 0
    if corrupt_dir.exists():
        corrupt_count = sum(1 for p in corrupt_dir.iterdir() if p.is_file())

    temp_parts = 0
    if temp_dir.exists():
        temp_parts = sum(1 for p in temp_dir.glob("*.part") if p.is_file())

    return {
        "image_count": image_count,
        "image_bytes": image_bytes,
        "corrupt_count": corrupt_count,
        "temp_parts": temp_parts,
    }


def format_bytes(size: int) -> str:
    value = float(size or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TB"


@st.cache_data(ttl=10)
def latest_mega_log_snapshot(data_dir_str: str) -> dict:
    temp_dir = Path(data_dir_str) / "temp"
    logs = list(temp_dir.glob("mega_sync*.err.log")) if temp_dir.exists() else []
    if not logs:
        return {"path": "", "last_line": "", "updated_at": None}
    latest = max(logs, key=lambda p: p.stat().st_mtime)
    try:
        lines = latest.read_text(encoding="utf-8", errors="replace").splitlines()
        last_line = lines[-1] if lines else ""
    except Exception as exc:
        last_line = str(exc)
    return {"path": str(latest), "last_line": last_line, "updated_at": latest.stat().st_mtime}


def build_archive_health(store: ChromaStore) -> dict:
    settings = get_settings()
    data_dir = settings.get_data_path()
    state = read_json_cached(str(data_dir / "sync_state.json"), {})
    failed = read_json_cached(str(data_dir / "mega_failed.json"), {"items": {}})
    report = read_json_cached(str(data_dir / "audit_report.json"), {})
    file_stats = scan_archive_files(str(data_dir))
    log_snapshot = latest_mega_log_snapshot(str(data_dir))

    channels = state.get("channels", {}) if isinstance(state.get("channels"), dict) else {}
    audit_channels = [
        item for item in channels.values()
        if isinstance(item, dict) and ("audit_cursor_id" in item or item.get("audit_done"))
    ]
    failed_items = failed.get("items", {}) if isinstance(failed.get("items"), dict) else {}
    failed_by_stage = Counter(str(item.get("stage") or "unknown") for item in failed_items.values() if isinstance(item, dict))
    failed_by_error = Counter(str(item.get("error") or "unknown")[:100] for item in failed_items.values() if isinstance(item, dict))

    try:
        chroma_count = store.get_count()
    except Exception:
        chroma_count = 0

    audit_total = len(audit_channels)
    audit_done = sum(1 for item in audit_channels if item.get("audit_done"))
    audit_scanned = sum(int(item.get("audit_scanned") or 0) for item in audit_channels)

    return {
        "collection": settings.get_collection_name(),
        "model": settings.clip_model_name,
        "chroma_count": chroma_count,
        "image_count": file_stats["image_count"],
        "image_bytes": file_stats["image_bytes"],
        "corrupt_count": file_stats["corrupt_count"],
        "temp_parts": file_stats["temp_parts"],
        "failed_count": len(failed_items),
        "failed_by_stage": failed_by_stage.most_common(8),
        "failed_by_error": failed_by_error.most_common(8),
        "audit_total": audit_total,
        "audit_done": audit_done,
        "audit_scanned": audit_scanned,
        "latest_log": log_snapshot.get("path") or report.get("latest_log", ""),
        "last_log_line": log_snapshot.get("last_line") or report.get("last_log_line", ""),
        "report_generated_at": report.get("generated_at", ""),
    }


@st.cache_data(ttl=30)
def load_quality_reports(data_dir_str: str) -> dict:
    reports_dir = Path(data_dir_str) / "reports"
    files = {
        "cleanvision": reports_dir / "image_quality_report.json",
        "duplicates": reports_dir / "duplicates_candidates.json",
        "ram": reports_dir / "ram_tagging_report.json",
    }
    loaded = {}
    for key, path in files.items():
        if not path.exists():
            loaded[key] = {"exists": False, "path": str(path)}
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data["exists"] = True
                data["path"] = str(path)
                loaded[key] = data
            else:
                loaded[key] = {"exists": False, "path": str(path), "error": "invalid json root"}
        except Exception as exc:
            loaded[key] = {"exists": False, "path": str(path), "error": str(exc)}
    return loaded


def run_health_sample(store: ChromaStore, sample_size: int) -> dict:
    settings = get_settings()
    images_dir = settings.get_images_path()
    sample_size = max(1, int(sample_size))
    missing_files = []
    files_missing_db = []
    checked_db = 0
    checked_files = 0

    try:
        result = store.collection.get(limit=sample_size, include=["metadatas"])
        ids = result.get("ids", []) if result else []
        metas = result.get("metadatas", []) if result else []
        checked_db = len(ids)
        for doc_id, meta in zip(ids, metas):
            meta = meta or {}
            image_path = meta.get("image_path") or meta.get("file_path") or str(images_dir / f"{doc_id}.jpg")
            if not Path(image_path).exists():
                missing_files.append({"id": doc_id, "image_path": image_path})
    except Exception as exc:
        missing_files.append({"id": "sample_error", "image_path": str(exc)})

    try:
        image_files = []
        for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            image_files.extend(images_dir.glob(pattern))
            if len(image_files) >= sample_size:
                break
        image_files = image_files[:sample_size]
        checked_files = len(image_files)
        ids = [p.stem for p in image_files]
        existing = set()
        if ids:
            try:
                result = store.collection.get(ids=ids, include=[])
                existing = set(result.get("ids", [])) if result else set()
            except Exception:
                existing = set()
        for file_path in image_files:
            if file_path.stem not in existing:
                files_missing_db.append({"id": file_path.stem, "image_path": str(file_path)})
    except Exception as exc:
        files_missing_db.append({"id": "sample_error", "image_path": str(exc)})

    return {
        "checked_db": checked_db,
        "checked_files": checked_files,
        "missing_files": missing_files[:50],
        "files_missing_db": files_missing_db[:50],
    }


# ══════════════════════════════════════════════════
# Başlık
# ══════════════════════════════════════════════════



def command_result(args: list[str], timeout: int = 60) -> dict:
    started = datetime.now().isoformat(timespec="seconds")
    try:
        completed = subprocess.run(
            [sys.executable, *args],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "started_at": started,
            "returncode": completed.returncode,
            "stdout": completed.stdout or "",
            "stderr": completed.stderr or "",
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "started_at": started,
            "returncode": -1,
            "stdout": exc.stdout or "",
            "stderr": f"Timeout after {timeout}s",
        }
    except Exception as exc:
        return {"started_at": started, "returncode": -1, "stdout": "", "stderr": str(exc)}


def start_background_job(name: str, args: list[str]) -> dict:
    safe_name = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in name).strip("_") or "job"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = get_settings().get_data_path() / "temp" / "ui_jobs"
    temp_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = temp_dir / f"{safe_name}_{timestamp}.out.log"
    stderr_path = temp_dir / f"{safe_name}_{timestamp}.err.log"
    stdout_file = open(stdout_path, "w", encoding="utf-8")
    stderr_file = open(stderr_path, "w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            [sys.executable, "-u", *args],
            cwd=str(PROJECT_ROOT),
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
        return {
            "ok": True,
            "pid": process.pid,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "command": " ".join([sys.executable, "-u", *args]),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "stdout": str(stdout_path), "stderr": str(stderr_path)}
    finally:
        stdout_file.close()
        stderr_file.close()


def tail_text(path_str: str, lines: int = 40) -> str:
    path = Path(path_str)
    if not path.exists():
        return ""
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except Exception as exc:
        return str(exc)


def show_command_output(result: dict) -> None:
    status = "OK" if result.get("returncode") == 0 else "HATA"
    st.caption(f"Durum: {status} | Kod: {result.get('returncode')} | Baslangic: {result.get('started_at')}")
    if result.get("stdout"):
        st.code(result["stdout"], language="text")
    if result.get("stderr"):
        st.code(result["stderr"], language="text")


def render_report_images(paths: list[str], max_items: int = 12) -> None:
    clean_paths = [p for p in paths if p]
    if not clean_paths:
        st.info("Gosterilecek gorsel yok.")
        return
    columns = st.columns(4)
    for idx, raw_path in enumerate(clean_paths[:max_items]):
        path = Path(raw_path)
        if not path.is_absolute():
            path = get_settings().get_images_path() / raw_path
        with columns[idx % 4]:
            if path.exists():
                st.image(str(path), use_container_width=True)
                st.caption(path.name)
            else:
                st.warning(path.name)


st.markdown('<h1 class="main-title">🔍 3D Model Arama Motoru</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Telegram kanallarından toplanan 3D modelleri AI ile arayın</p>',
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════
# Sidebar — İstatistikler ve Filtreler
# ══════════════════════════════════════════════════

with st.sidebar:
    # ── Sidebar: Sistem Bilgisi ──
    st.markdown('<div class="sidebar-section-header">🤖 SİSTEM BİLGİSİ</div>', unsafe_allow_html=True)
    settings = get_settings()
    st.caption(f"🧠 Model: `{settings.clip_model_name}`")
    st.caption(f"📦 Koleksiyon: `{settings.get_collection_name()}`")
    
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # ── Sidebar: Sistem Durumu ──
    st.markdown('<div class="sidebar-section-header">📊 SİSTEM DURUMU</div>', unsafe_allow_html=True)
    
    # İstatistikleri hesapla (güvenli)
    try:
        total_models = store.get_count()
    except Exception:
        total_models = 0
    images_dir = Path("./data/images")
    img_count = len(list(images_dir.glob("*.jpg"))) if images_dir.exists() else 0
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f'<div class="side-metric-card">'
            f'<div class="side-metric-value">{total_models:,}</div>'
            f'<div class="side-metric-label">İndeksli Model</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f'<div class="side-metric-card">'
            f'<div class="side-metric-value">{img_count:,}</div>'
            f'<div class="side-metric-label">İndirilen Görsel</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # ── Sidebar: Arama Ayarları ──
    st.markdown('<div class="sidebar-section-header">⚙️ ARAMA AYARLARI</div>', unsafe_allow_html=True)
    
    show_all = st.checkbox("🔓 Tümünü Göster", value=False, key="show_all_checkbox", help="Benzerlik skoru düşük olanları da getirir.")
    
    max_results = max(total_models, 100)
    if show_all:
        n_results = max_results
        st.info(f"Tüm arşiv gösteriliyor ({max_results} model)")
    else:
        n_results = st.slider("Sonuç Sayısı", 10, min(max_results, 500), 100, step=10, key="n_results_slider")

    min_score = st.slider(
        "Minimum Benzerlik Skoru", 0.0, 1.0, 0.05,
        step=0.01, key="min_score_slider",
        help="0.0: Her şeyi getirir, 1.0: Sadece birebir aynılarını getirir."
    )

    min_aesthetic = st.slider(
        "✨ Minimum Estetik Skor", 0.0, 10.0, 0.0,
        step=0.5, key="min_aesthetic_slider",
        help="0: Filtre yok · 4+: Orta · 6+: İyi · 8+: Elit kalite"
    )
    if min_aesthetic > 0:
        tier_label = "🏆 Elit" if min_aesthetic >= 8 else "🌟 İyi" if min_aesthetic >= 6 else "👍 Orta" if min_aesthetic >= 4 else "📋 Tümü"
        st.caption(f"Aktif filtre: **{tier_label}** ({min_aesthetic:.1f}+)")

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # ── Sidebar: Semantik Filtreler (SigLIP 2 Taksonomisi) ──
    st.markdown('<div class="sidebar-section-header">🧪 SEMANTİK FİLTRELER</div>', unsafe_allow_html=True)
    
    # Kategori Filtresi
    categories = ["sofa", "armchair", "dining chair", "office chair", "dining table", "coffee table", "side table", "desk", "console table", "bed", "wardrobe", "dresser", "nightstand", "bookshelf", "cabinet", "sideboard", "tv unit", "chandelier", "pendant light", "floor lamp", "table lamp", "wall sconce", "mirror", "rug", "curtain", "plant", "vase", "sculpture", "painting", "bathtub", "sink", "fireplace"]
    selected_cats = st.multiselect("🏷️ Mobilya Tipi", categories, help="SigLIP 2 ile tespit edilen mobilya tipleri")
    
    # Stil Filtresi
    styles = ["modern", "minimalist", "contemporary", "scandinavian", "japandi", "industrial", "loft", "rustic", "mid-century modern", "classic", "neoclassic", "baroque", "provence", "art deco", "bohemian", "luxury", "retro", "vintage", "futuristic"]
    selected_styles = st.multiselect("🎭 Stiller", styles, help="SigLIP 2 ile tespit edilen stiller")
    
    # Materyal Filtresi
    materials = ["wood", "walnut", "oak", "pine", "metal", "steel", "brass", "chrome", "glass", "marble", "granite", "concrete", "stone", "leather", "fabric", "velvet", "boucle", "linen", "ceramic", "porcelain", "rattan", "bamboo", "plastic"]
    selected_mats = st.multiselect("🧱 Malzemeler", materials, help="SigLIP 2 ile tespit edilen malzemeler")
    
    # Renk Filtresi
    colors = ["white", "black", "gray", "anthracite", "beige", "cream", "brown", "walnut brown", "blue", "navy blue", "green", "olive green", "red", "burgundy", "terracotta", "yellow", "mustard yellow", "gold color", "silver color", "brass color", "pink", "orange"]
    selected_colors = st.multiselect("🎨 Renkler", colors, help="SigLIP 2 ile tespit edilen renkler")
    
    # Oda Filtresi
    rooms = ["living room", "bedroom", "bathroom", "kitchen", "dining room", "office", "study room", "hallway", "entryway", "kids room", "dressing room", "outdoor", "garden", "balcony", "commercial space"]
    selected_rooms = st.multiselect("🏠 Oda Tipi", rooms, help="SigLIP 2 ile tespit edilen oda tipleri")

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # ── Sidebar: Sayfalama ──
    st.markdown('<div class="sidebar-section-header">📄 SAYFALAMA</div>', unsafe_allow_html=True)
    per_page = st.select_slider(
        "Sayfa başına sonuç",
        options=[12, 24, 48, 72, 96],
        value=24,
        key="per_page_slider",
    )

    st.markdown("---")
    st.markdown("### 📖 Kullanım")
    st.markdown("""
    **Metin Araması**: "ahşap duvar paneli" gibi doğal dilde yazın.

    **Görsel Araması**: Pinterest'ten indirdiğiniz referans fotoğrafı yükleyin.

    **Telegram'a Git**: Beğendiğiniz sonuca tıklayarak direkt modele ulaşın.
    """)


# ══════════════════════════════════════════════════
# Ana Arama Alanı ve Sekmeler
# ══════════════════════════════════════════════════

tab_search, tab_scene, tab_health, tab_quality, tab_jobs, tab_chat = st.tabs(["🔍 Arama Motoru", "📸 Sahne Analizi", "🏥 Arsiv Sagligi", "🧪 Kalite Lab", "⚙️ Gorev Merkezi", "💬 Tasarim Asistani"])

with tab_search:
    # Arama modu seçimi
    search_mode_labels = ["📝 Metin ile Ara", "🖼️ Görsel ile Ara"]
    search_mode_index = st.session_state.get("search_mode", 0)
    if not isinstance(search_mode_index, int) or search_mode_index not in range(len(search_mode_labels)):
        search_mode_index = 0
        st.session_state.search_mode = search_mode_index
    if st.session_state.get("search_mode_radio") != search_mode_labels[search_mode_index]:
        st.session_state.search_mode_radio = search_mode_labels[search_mode_index]
    
    search_mode = st.radio(
        "Arama Modu",
        search_mode_labels,
        index=search_mode_index,
        horizontal=True,
        key="search_mode_radio",
    )
    # State'i güncelle
    st.session_state.search_mode = search_mode_labels.index(search_mode)

    results = []

    if search_mode == "📝 Metin ile Ara":
        # ── Metin Araması ──
        query = st.text_input(
            "Arama sorgusu",
            placeholder="Ör: ahşap çıtalı duvar paneli, minimalist sehpa, mermer zemin...",
            key="text_search_input",
            label_visibility="collapsed",
        )

        if query and query.strip():
            query_key = f"text:{query.strip()}:{n_results}:{min_score}"
            if query_key != st.session_state.last_query_key:
                st.session_state.current_page = 0
                with st.spinner("🧠 AI Sorguyu Genişletiyor ve CLIP ile Aranıyor..."):
                    res, expanded_query = search_by_text(
                        query=query,
                        store=store,
                        n_results=n_results,
                        min_score=min_score,
                    )
                    st.session_state.search_results = res
                    st.session_state.expanded_query = expanded_query
                st.session_state.last_query_key = query_key
            
            results = st.session_state.search_results
            expanded = st.session_state.get("expanded_query", query)
            
            if expanded != query:
                st.markdown(
                    f"<div style='background:rgba(102, 126, 234, 0.1); border-left:4px solid #667eea; padding:10px; border-radius:4px; margin-bottom:15px;'>"
                    f"<small style='color:#8899aa;'>✨ AI Genişletilmiş Arama:</small><br>"
                    f"<b>{expanded}</b></div>",
                    unsafe_allow_html=True
                )

    else:
        # ── Görsel Araması ──
        target_image_path = None
        
        # Eğer kullanıcı bir sonuç kartından "Benzerlerini Bul"a bastıysa
        if st.session_state.get("target_similar_image"):
            target_image_path = st.session_state.target_similar_image
            
            col_preview, col_info = st.columns([1, 2])
            with col_preview:
                st.image(target_image_path, caption="Arşivden Seçilen Referans", width=250)
            with col_info:
                st.info("🎯 Arama Modu: Arşivdeki bu görselin benzerleri aranıyor...")
                if st.button("❌ İptal Et ve Yeni Görsel Yükle"):
                    st.session_state.target_similar_image = None
                    st.rerun()
        else:
            uploaded_file = st.file_uploader(
                "Referans fotoğrafı yükleyin (Pinterest, Google vb.)",
                type=["jpg", "jpeg", "png", "webp"],
                key="image_upload",
            )

            if uploaded_file is not None:
                col_preview, col_info = st.columns([1, 2])
                with col_preview:
                    st.image(uploaded_file, caption="Yuklenen Referans", width=250)

                # 🛡️ GÜVENLİK: Benzersiz dosya ismi (Cache Fix)
                import hashlib
                file_hash = hashlib.md5(uploaded_file.getbuffer()).hexdigest()
                temp_path = Path("./data/temp") / f"query_{file_hash}.jpg"
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                target_image_path = str(temp_path)
                file_size = uploaded_file.size
            else:
                file_size = 0

        if target_image_path:
            # key oluştururken dinamik olması için image path ve size
            query_key = f"image:{target_image_path}:{n_results}:{min_score}"
            if query_key != st.session_state.last_query_key:
                st.session_state.current_page = 0
                with st.spinner("🧠 Benzer modeller ve AI Görsel Analizi yapılıyor..."):
                    res, analysis_text = search_by_image(
                        image_path=target_image_path,
                        store=store,
                        n_results=n_results,
                        min_score=min_score,
                    )
                    st.session_state.search_results = res
                    st.session_state.analysis_text = analysis_text
                st.session_state.last_query_key = query_key
                
            results = st.session_state.search_results
            analysis = st.session_state.get("analysis_text", "")
            
            if analysis:
                st.markdown(
                    f"<div style='background:rgba(102, 126, 234, 0.1); border-left:4px solid #667eea; padding:10px; border-radius:4px; margin-bottom:15px; margin-top:15px;'>"
                    f"<small style='color:#8899aa;'>✨ AI Görsel Analizi (Tarz & Materyal):</small><br>"
                    f"<b>{analysis}</b></div>",
                    unsafe_allow_html=True
                )


    # ══════════════════════════════════════════════════
    # Filtreleme Mantığı (Semantic Filtering)
    # ══════════════════════════════════════════════════
    
    if results:
        from config import MetadataSchema
        filtered_results = []
        for res in results:
            if not res: continue
            meta = res.get("metadata") or {}
            
            # SigLIP 2 etiketlerini oku (clip_ prefix)
            clip_tags_all = " ".join(str(meta.get(k, "")) for k in [
                "clip_furniture_type", "clip_style", "clip_material",
                "clip_color", "clip_room", "clip_category", "clip_tags",
                MetadataSchema.OBJECT_TYPE, MetadataSchema.STYLE,
                MetadataSchema.MATERIAL, MetadataSchema.ROOM,
                MetadataSchema.COLOR_FAMILY,
            ]).lower()
                
            # Filtre kontrolü — SigLIP 2 etiketleri üzerinden
            match_cat = not selected_cats or any(cat in clip_tags_all for cat in selected_cats)
            match_style = not selected_styles or any(style in clip_tags_all for style in selected_styles)
            match_mat = not selected_mats or any(mat in clip_tags_all for mat in selected_mats)
            match_color = not selected_colors or any(color in clip_tags_all for color in selected_colors)
            match_room = not selected_rooms or any(room in clip_tags_all for room in selected_rooms)
            
            # Estetik skor filtresi
            aes = float(meta.get("aesthetic_score", 0) or 0)
            match_aesthetic = min_aesthetic <= 0 or aes >= min_aesthetic
            
            if match_cat and match_style and match_mat and match_color and match_room and match_aesthetic:
                filtered_results.append(res)
        
        results = filtered_results

    # ══════════════════════════════════════════════════
    # Sonuç Galerisi
    # ══════════════════════════════════════════════════

    if results:
        total_results = len(results)
        items_per_page = per_page
        total_pages = max(1, -(-total_results // items_per_page))

        if st.session_state.current_page >= total_pages:
            st.session_state.current_page = 0

        current_page = st.session_state.current_page
        start_idx = current_page * items_per_page
        end_idx = min(start_idx + items_per_page, total_results)
        page_results = results[start_idx:end_idx]

        st.markdown(
            f"### 🎯 {total_results} Sonuç Bulundu &nbsp;·&nbsp; "
            f"<span style='color:#8899aa;font-size:0.9rem;'>Sayfa {current_page + 1}/{total_pages}</span>",
            unsafe_allow_html=True,
        )

        if total_pages > 1:
            nav_cols = st.columns([1, 1, 2, 1, 1])
            with nav_cols[0]:
                if st.button("⏮ İlk", key="first_top", disabled=(current_page == 0)):
                    st.session_state.current_page = 0
                    st.rerun()
            with nav_cols[1]:
                if st.button("◀ Önceki", key="prev_top", disabled=(current_page == 0)):
                    st.session_state.current_page -= 1
                    st.rerun()
            with nav_cols[2]:
                st.markdown(
                    f"<div style='text-align:center;padding:8px;color:#8899aa;'>"
                    f"Gösterilen: {start_idx + 1}–{end_idx} / {total_results}</div>",
                    unsafe_allow_html=True,
                )
            with nav_cols[3]:
                if st.button("Sonraki ▶", key="next_top", disabled=(current_page >= total_pages - 1)):
                    st.session_state.current_page += 1
                    st.rerun()
            with nav_cols[4]:
                if st.button("Son ⏭", key="last_top", disabled=(current_page >= total_pages - 1)):
                    st.session_state.current_page = total_pages - 1
                    st.rerun()

        st.markdown("---")

        cols_per_row = 4
        for row_start in range(0, len(page_results), cols_per_row):
            cols = st.columns(cols_per_row)
            for col_idx, result in enumerate(page_results[row_start : row_start + cols_per_row]):
                with cols[col_idx]:
                    meta = result.get("metadata", {})
                    score = result.get("score", 0)
                    # Hem eski hem yeni anahtarları kontrol et (Uyumluluk için)
                    image_path = meta.get("image_path", meta.get("file_path", ""))
                    deep_link = meta.get("deep_link", "")
                    channel_title = meta.get("channel_title", "")

                    st.markdown('<div class="result-card">', unsafe_allow_html=True)

                    if image_path and Path(image_path).exists():
                        st.image(image_path, use_container_width=True)
                    else:
                        st.markdown("🖼️ *Görsel bulunamadı*")

                    score_pct = int(score * 100)
                    
                    # Estetik skor badge
                    aes_score = meta.get("aesthetic_score", 0)
                    aes_html = ""
                    if aes_score and float(aes_score) > 0:
                        aes_val = float(aes_score)
                        if aes_val >= 7.0:
                            aes_class = "aesthetic-elite"
                            aes_icon = "🏆"
                        elif aes_val >= 5.5:
                            aes_class = "aesthetic-good"
                            aes_icon = "✨"
                        elif aes_val >= 4.0:
                            aes_class = "aesthetic-mid"
                            aes_icon = "📷"
                        else:
                            aes_class = "aesthetic-low"
                            aes_icon = "📉"
                        aes_html = f'<span class="aesthetic-badge {aes_class}">{aes_icon} {aes_val:.1f}</span>'
                    
                    # SigLIP 2 etiketlerini oku
                    ai_tags_html = ""
                    clip_fields = ["clip_furniture_type", "clip_style", "clip_material", "clip_color"]
                    for field in clip_fields:
                        val = meta.get(field, "")
                        if val:
                            first_tag = str(val).split(",")[0].strip()
                            if first_tag:
                                ai_tags_html += f'<span class="ai-tag">{first_tag}</span>'
                    
                    # Fallback: eski etiketler
                    if not ai_tags_html:
                        for key in ["object_type", "style", "material", "room"]:
                            t = meta.get(key, "")
                            if t:
                                ai_tags_html += f'<span class="ai-tag">{str(t).replace("_", " ")}</span>'
                    
                    ch_display = channel_title[:18] if channel_title else "?"
                    st.markdown(f"""
                        <div class="badge-container">
                            <span class="score-badge">%{score_pct}</span>
                            {aes_html}
                            <span class="channel-tag">📡 {ch_display}</span>
                        </div>
                        <div class="badge-container" style="margin-top:4px;">{ai_tags_html}</div>
                    """, unsafe_allow_html=True)
                    
                    st.write("") # Küçük boşluk

                    # Butonları yan yana dizelim (Kompakt Mod)
                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        if deep_link:
                            st.link_button("📲 Git", deep_link, use_container_width=True, help="Telegram'da Aç")
                    with btn_col2:
                        if image_path and Path(image_path).exists():
                            if st.button("🔍 Bul", key=f"sim_{image_path}_{row_start}_{col_idx}", use_container_width=True, help="Benzerlerini Bul"):
                                st.session_state.target_similar_image = image_path
                                st.session_state.search_mode = 1  # 1: Görsel Araması İndeksi
                                st.session_state.last_query_key = ""
                                st.rerun()

                    st.markdown('</div>', unsafe_allow_html=True)

        if total_pages > 1:
            st.markdown("---")
            bot_cols = st.columns([1, 1, 2, 1, 1])
            with bot_cols[0]:
                if st.button("⏮ İlk", key="first_bot", disabled=(current_page == 0)):
                    st.session_state.current_page = 0
                    st.rerun()
            with bot_cols[1]:
                if st.button("◀ Önceki", key="prev_bot", disabled=(current_page == 0)):
                    st.session_state.current_page -= 1
                    st.rerun()
            with bot_cols[2]:
                st.markdown(
                    f"<div style='text-align:center;padding:8px;color:#8899aa;'>"
                    f"Sayfa {current_page + 1} / {total_pages}</div>",
                    unsafe_allow_html=True,
                )
            with bot_cols[3]:
                if st.button("Sonraki ▶", key="next_bot", disabled=(current_page >= total_pages - 1)):
                    st.session_state.current_page += 1
                    st.rerun()
            with bot_cols[4]:
                if st.button("Son ⏭", key="last_bot", disabled=(current_page >= total_pages - 1)):
                    st.session_state.current_page = total_pages - 1
                    st.rerun()

    elif (search_mode == "📝 Metin ile Ara" and query) or \
         (search_mode == "🖼️ Görsel ile Ara" and target_image_path):
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-state-emoji">🔍</div>'
            '<h3>Sonuç Bulunamadı</h3>'
            '<p>Farklı anahtar kelimeler deneyin veya minimum skor eşiğini düşürün.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-state-emoji">🏛️</div>'
            '<h3>3D Model Arama Motoruna Hoş Geldiniz</h3>'
            '<p>Yukarıdaki arama çubuğuna metin yazın veya referans görsel yükleyin.</p>'
            '</div>',
            unsafe_allow_html=True,
        )


with tab_scene:
    st.subheader("📸 Sahne Analizi — Pinterest Fotoğrafından Obje Tespiti")
    st.caption("Bir oda fotoğrafı yükleyin → OWLv2 ile objeleri tespit edelim → Her obje için benzer 3D modeller bulalım.")
    
    scene_file = st.file_uploader(
        "Oda/Sahne fotoğrafı yükleyin",
        type=["jpg", "jpeg", "png", "webp"],
        key="scene_upload",
    )
    
    if scene_file is not None:
        import hashlib
        scene_hash = hashlib.md5(scene_file.getbuffer()).hexdigest()
        scene_path = Path("./data/temp") / f"scene_{scene_hash}.jpg"
        scene_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(scene_path, "wb") as f:
            f.write(scene_file.getbuffer())
        
        st.image(str(scene_path), caption="Yüklenen Sahne", use_container_width=True)
        
        confidence = st.slider("Tespit Güven Eşiği", 0.05, 0.50, 0.12, step=0.01, key="scene_confidence")
        n_models = st.slider("Obje başına sonuç", 4, 16, 8, key="scene_n_models")
        
        if st.button("🔍 Sahneyi Analiz Et", use_container_width=True, type="primary"):
            with st.spinner("🦉 OWLv2 ile objeler tespit ediliyor..."):
                try:
                    from search.scene_parser import SceneParser
                    parser = SceneParser(confidence_threshold=confidence)
                    
                    detections = parser.detect_objects(str(scene_path))
                    if not detections:
                        st.warning("Hiçbir obje tespit edilemedi. Güven eşiğini düşürmeyi deneyin.")
                    else:
                        st.success(f"🎯 {len(detections)} obje tespit edildi!")
                        
                        detections = parser.crop_objects(str(scene_path), detections)
                        
                        with st.spinner("🔍 Her obje için benzer 3D modeller aranıyor..."):
                            detections = parser.search_similar_models(detections, n_results=n_models)
                        
                        st.session_state.scene_detections = detections
                    
                    parser.unload()
                except Exception as e:
                    st.error(f"Sahne analizi hatası: {e}")
        
        # Sonuçları göster
        if st.session_state.get("scene_detections"):
            for det in st.session_state.scene_detections:
                with st.expander(f"🏷️ {det['label'].upper()} — Güven: {det['confidence']:.0%}", expanded=True):
                    col_crop, col_models = st.columns([1, 3])
                    
                    with col_crop:
                        crop_path = det.get("crop_path", "")
                        if crop_path and Path(crop_path).exists():
                            st.image(crop_path, caption=f"Tespit: {det['label']}", use_container_width=True)
                        st.caption(f"Konum: {det['bbox']}")
                    
                    with col_models:
                        similar = det.get("similar_models", [])
                        if similar:
                            model_cols = st.columns(min(4, len(similar)))
                            for mi, model in enumerate(similar[:8]):
                                with model_cols[mi % 4]:
                                    m_meta = model.get("metadata", {})
                                    m_path = m_meta.get("image_path", "")
                                    m_score = model.get("score", 0)
                                    m_link = m_meta.get("deep_link", "")
                                    
                                    if m_path and Path(m_path).exists():
                                        st.image(m_path, use_container_width=True)
                                    st.caption(f"Benzerlik: {int(m_score*100)}%")
                                    if m_link:
                                        st.link_button("📲 Git", m_link, use_container_width=True)
                        else:
                            st.info("Bu obje için benzer model bulunamadı.")


with tab_health:
    st.subheader("Arsiv Sagligi")
    st.caption("Bu ekran DB, dosya, audit ve failed durumunu okur; veri silmez veya tasimaz.")

    refresh_col, sample_col = st.columns([1, 3])
    with refresh_col:
        if st.button("Yenile", use_container_width=True):
            read_json_cached.clear()
            scan_archive_files.clear()
            latest_mega_log_snapshot.clear()
            load_quality_reports.clear()
            st.rerun()

    health = build_archive_health(store)
    diff = health["image_count"] - health["chroma_count"]

    metric_cols = st.columns(4)
    metric_cols[0].metric("Chroma Kaydi", f"{health['chroma_count']:,}")
    metric_cols[1].metric("Gorsel Dosyasi", f"{health['image_count']:,}", delta=f"{diff:+,} dosya farki")
    metric_cols[2].metric("Failed", f"{health['failed_count']:,}")
    metric_cols[3].metric("Karantina", f"{health['corrupt_count']:,}")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Audit Kanal", f"{health['audit_done']:,}/{health['audit_total']:,}")
    metric_cols[1].metric("Taranan Foto", f"{health['audit_scanned']:,}")
    metric_cols[2].metric("Gecici Part", f"{health['temp_parts']:,}")
    metric_cols[3].metric("Disk Boyutu", format_bytes(health["image_bytes"]))

    if health["audit_total"]:
        st.progress(min(1.0, health["audit_done"] / max(1, health["audit_total"])))

    left, right = st.columns(2)
    with left:
        st.markdown("#### Failed Ozeti")
        if health["failed_by_stage"]:
            st.dataframe(
                [{"stage": stage, "count": count} for stage, count in health["failed_by_stage"]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success("Failed listesi bos gorunuyor.")

    with right:
        st.markdown("#### Son Log")
        st.caption(health.get("latest_log") or "Log bulunamadi")
        st.code(health.get("last_log_line") or "Son log satiri yok", language="text")

    with st.expander("En sik failed hatalari", expanded=False):
        if health["failed_by_error"]:
            st.dataframe(
                [{"error": error, "count": count} for error, count in health["failed_by_error"]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.write("Kayit yok.")

    st.markdown("#### Kalite Raporlari")
    quality_reports = load_quality_reports(str(get_settings().get_data_path()))
    q_cols = st.columns(3)

    cleanvision = quality_reports.get("cleanvision", {})
    with q_cols[0]:
        if cleanvision.get("exists"):
            issue_total = sum(int(item.get("num_images") or 0) for item in cleanvision.get("issue_summary", []))
            st.metric("CleanVision issue", f"{issue_total:,}")
            st.caption(f"Kontrol edilen: {int(cleanvision.get('checked_images') or 0):,}")
        else:
            st.metric("CleanVision issue", "rapor yok")

    duplicates_report = quality_reports.get("duplicates", {})
    with q_cols[1]:
        if duplicates_report.get("exists"):
            st.metric("Duplicate grup", f"{int(duplicates_report.get('group_count') or 0):,}")
            st.caption(f"Pair: {int(duplicates_report.get('pair_count') or 0):,}")
        else:
            st.metric("Duplicate grup", "rapor yok")

    ram_report = quality_reports.get("ram", {})
    with q_cols[2]:
        if ram_report.get("exists"):
            st.metric("RAM etiketli", f"{int(ram_report.get('tagged_images') or 0):,}")
            st.caption(f"Model: {ram_report.get('model', '-')}")
        else:
            st.metric("RAM etiketli", "rapor yok")

    with st.expander("Kalite raporu detaylari", expanded=False):
        if cleanvision.get("exists") and cleanvision.get("issue_summary"):
            st.markdown("##### CleanVision")
            st.dataframe(cleanvision.get("issue_summary", []), use_container_width=True, hide_index=True)
        if duplicates_report.get("exists") and duplicates_report.get("groups"):
            st.markdown("##### Duplicate aday gruplari")
            st.dataframe(duplicates_report.get("groups", [])[:50], use_container_width=True, hide_index=True)
        if ram_report.get("exists") and ram_report.get("records"):
            st.markdown("##### Recognize Anything ornekleri")
            st.dataframe(ram_report.get("records", [])[:50], use_container_width=True, hide_index=True)
        if not any(report.get("exists") for report in quality_reports.values()):
            st.write("Henuz kalite raporu yok. `scripts/image_quality_audit.py`, `scripts/duplicate_candidates.py` veya `scripts/recognize_anything_poc.py` calistirilinca burada gorunur.")

    st.markdown("#### Orneklem Tutarlilik Kontrolu")
    sample_size = st.select_slider("Orneklem boyutu", options=[100, 250, 500, 1000, 2500, 5000], value=500)
    if st.button("Orneklem kontrolu calistir", use_container_width=True):
        with st.spinner("DB ve dosya orneklemi karsilastiriliyor..."):
            sample = run_health_sample(store, sample_size)
        sample_cols = st.columns(4)
        sample_cols[0].metric("DB Ornek", f"{sample['checked_db']:,}")
        sample_cols[1].metric("Eksik Dosya", f"{len(sample['missing_files']):,}")
        sample_cols[2].metric("Dosya Ornek", f"{sample['checked_files']:,}")
        sample_cols[3].metric("DB Eksik", f"{len(sample['files_missing_db']):,}")

        if sample["missing_files"]:
            st.markdown("##### Chroma var, dosya yok")
            st.dataframe(sample["missing_files"], use_container_width=True, hide_index=True)
        if sample["files_missing_db"]:
            st.markdown("##### Dosya var, Chroma yok")
            st.dataframe(sample["files_missing_db"], use_container_width=True, hide_index=True)
        if not sample["missing_files"] and not sample["files_missing_db"]:
            st.success("Orneklemde tutarsizlik bulunmadi.")


with tab_quality:
    st.subheader("Kalite Laboratuvari")
    st.caption("CleanVision, imagededup ve Recognize Anything raporlari burada incelenir. Bu sayfa veri silmez.")

    reports = load_quality_reports(str(get_settings().get_data_path()))
    cleanvision = reports.get("cleanvision", {})
    duplicates_report = reports.get("duplicates", {})
    ram_report = reports.get("ram", {})

    top_cols = st.columns(3)
    with top_cols[0]:
        if cleanvision.get("exists"):
            issue_total = sum(int(item.get("num_images") or 0) for item in cleanvision.get("issue_summary", []))
            st.metric("CleanVision issue", f"{issue_total:,}")
            st.caption(cleanvision.get("path", ""))
        else:
            st.metric("CleanVision issue", "rapor yok")
    with top_cols[1]:
        if duplicates_report.get("exists"):
            st.metric("Duplicate grup", f"{int(duplicates_report.get('group_count') or 0):,}")
            st.caption(f"Pair: {int(duplicates_report.get('pair_count') or 0):,}")
        else:
            st.metric("Duplicate grup", "rapor yok")
    with top_cols[2]:
        if ram_report.get("exists"):
            st.metric("RAM etiketli", f"{int(ram_report.get('tagged_images') or 0):,}")
            st.caption(f"Model: {ram_report.get('model', '-')}")
        else:
            st.metric("RAM etiketli", "rapor yok")

    q_tab1, q_tab2, q_tab3 = st.tabs(["CleanVision", "Duplicate Review", "RAM Tagging"])

    with q_tab1:
        if not cleanvision.get("exists"):
            st.info("CleanVision raporu yok. Gorev Merkezi'nden kalite audit baslatabilirsin.")
        else:
            st.dataframe(cleanvision.get("issue_summary", []), use_container_width=True, hide_index=True)
            examples = cleanvision.get("top_examples", {}) or {}
            if examples:
                issue_names = sorted(examples.keys())
                selected_issue = st.selectbox("Issue tipi", issue_names, key="quality_issue_select")
                example_paths = [item.get("image_path", "") for item in examples.get(selected_issue, [])]
                render_report_images(example_paths, max_items=12)

    with q_tab2:
        if not duplicates_report.get("exists"):
            st.info("Duplicate raporu yok. Gorev Merkezi'nden imagededup analizi baslatabilirsin.")
        else:
            groups = duplicates_report.get("groups", []) or []
            st.dataframe(groups[:200], use_container_width=True, hide_index=True)
            if groups:
                options = [f"Grup {item.get('group_id')} ({item.get('count')} dosya)" for item in groups]
                selected_label = st.selectbox("Gorsel grup incele", options, key="duplicate_group_select")
                selected_idx = options.index(selected_label)
                selected_group = groups[selected_idx]
                render_report_images(selected_group.get("files", []), max_items=16)
                st.caption("Bu ekran sadece adaylari gosterir; silme/tasima islemi yapmaz.")

    with q_tab3:
        if not ram_report.get("exists"):
            st.info("Recognize Anything POC raporu yok. Repo ve checkpoint hazir olunca Gorev Merkezi'nden POC baslatabilirsin.")
        else:
            records = ram_report.get("records", []) or []
            st.dataframe(records[:200], use_container_width=True, hide_index=True)
            if records:
                names = [Path(item.get("image_path", f"item_{idx}")).name for idx, item in enumerate(records)]
                selected = st.selectbox("Tag ornegi", names, key="ram_record_select")
                selected_record = records[names.index(selected)]
                render_report_images([selected_record.get("image_path", "")], max_items=1)
                st.write(", ".join(selected_record.get("tags", [])) or "Tag bulunamadi")


with tab_jobs:
    st.subheader("Gorev Merkezi")
    st.caption("Buradan durum okunur ve rapor ureten guvenli isler baslatilir. Veri silme/tasima butonu yoktur.")

    status_cols = st.columns(3)
    with status_cols[0]:
        if st.button("MegaSync status", use_container_width=True):
            with st.spinner("MegaSync durumu okunuyor..."):
                st.session_state.job_status_result = command_result(["scripts/mega_sync.py", "--status"], timeout=45)
    with status_cols[1]:
        if st.button("Audit raporu yaz", use_container_width=True):
            with st.spinner("Audit raporu yaziliyor..."):
                st.session_state.job_status_result = command_result(["scripts/mega_sync.py", "--write-report"], timeout=60)
                read_json_cached.clear()
                latest_mega_log_snapshot.clear()
    with status_cols[2]:
        if st.button("Raporlari yenile", use_container_width=True):
            read_json_cached.clear()
            scan_archive_files.clear()
            latest_mega_log_snapshot.clear()
            load_quality_reports.clear()
            st.rerun()

    if st.session_state.get("job_status_result"):
        show_command_output(st.session_state.job_status_result)

    st.markdown("#### Kalite Isleri")
    job_cols = st.columns(3)
    with job_cols[0]:
        clean_sample = st.number_input("CleanVision sample", min_value=100, max_value=100000, value=5000, step=500)
        if st.button("CleanVision baslat", use_container_width=True):
            st.session_state.last_started_job = start_background_job(
                "cleanvision",
                ["scripts/image_quality_audit.py", "--sample", str(int(clean_sample))],
            )
    with job_cols[1]:
        dup_sample = st.number_input("Duplicate sample", min_value=100, max_value=200000, value=10000, step=1000)
        dup_threshold = st.number_input("PHash threshold", min_value=1, max_value=30, value=10, step=1)
        if st.button("Duplicate analizi baslat", use_container_width=True):
            st.session_state.last_started_job = start_background_job(
                "duplicates",
                ["scripts/duplicate_candidates.py", "--method", "phash", "--sample", str(int(dup_sample)), "--threshold", str(int(dup_threshold))],
            )
    with job_cols[2]:
        ram_sample = st.number_input("RAM sample", min_value=1, max_value=500, value=10, step=1)
        if st.button("RAM POC dry-run", use_container_width=True):
            st.session_state.job_status_result = command_result(["scripts/recognize_anything_poc.py", "--dry-run", "--sample", str(int(ram_sample))], timeout=30)

    if st.session_state.get("last_started_job"):
        job = st.session_state.last_started_job
        if job.get("ok"):
            st.success(f"Is baslatildi. PID: {job.get('pid')}")
            st.caption(job.get("command", ""))
            st.write("stdout", job.get("stdout"))
            st.write("stderr", job.get("stderr"))
            if st.button("Son logu goster", use_container_width=True):
                st.code(tail_text(job.get("stderr", "")) or tail_text(job.get("stdout", "")), language="text")
        else:
            st.error(job.get("error", "Is baslatilamadi"))

    st.markdown("#### Canli MegaSync Log")
    snapshot = latest_mega_log_snapshot(str(get_settings().get_data_path()))
    st.caption(snapshot.get("path") or "Log bulunamadi")
    if snapshot.get("path"):
        st.code(tail_text(snapshot["path"], lines=30), language="text")

with tab_chat:
    from search.chat_assistant import annotate_results, chat_with_designer_v2, explain_results, update_archi_memory
    
    # ── Premium Chat Header ──
    st.markdown("""
    <div style='text-align:center; padding: 2rem 1rem 1.5rem;'>
        <div style='font-size: 2.8rem; margin-bottom: 0.3rem;'>🤖</div>
        <h2 style='margin:0; font-size:1.6rem; font-weight:700; background: linear-gradient(135deg, #667eea, #764ba2, #f093fb); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Archi Tasarım Asistanı</h2>
        <p style='color:#667eea; font-size:0.82rem; margin-top:6px; font-weight:400; letter-spacing:0.3px;'>Powered by Gemini AI · 460K+ 3D Model Arşivi</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Görsel Yükleme (Collapsible) ──
    if "chat_uploader_key" not in st.session_state:
        st.session_state.chat_uploader_key = 0
    
    with st.expander("📎 Referans fotoğraf ekle", expanded=False):
        chat_uploaded_file = st.file_uploader(
            "Bir fotoğraf yükle", 
            type=["jpg", "jpeg", "png"], 
            key=f"chat_file_uploader_{st.session_state.chat_uploader_key}",
            label_visibility="collapsed"
        )

    # ── Sohbet Geçmişini Render Et ──
    for i, msg in enumerate(st.session_state.chat_history):
        avatar = "🤖" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            
            if msg.get("image"):
                st.image(msg["image"], width=250)
            
            if msg.get("results"):
                result_count = len(msg["results"])
                st.markdown(f'<div style="margin:12px 0 8px; padding:8px 12px; background:rgba(102,126,234,0.08); border-radius:10px; border-left:3px solid #667eea; font-size:0.82rem; color:#8eaaff; font-weight:600;">✨ {result_count} model bulundu</div>', unsafe_allow_html=True)
                cols = st.columns(3)
                for idx, res in enumerate(msg["results"][:9]):
                    with cols[idx % 3]:
                        meta = res.get("metadata", {})
                        img_path = meta.get("image_path", "")
                        link = meta.get("deep_link", "")
                        if img_path and Path(img_path).exists():
                            st.image(img_path, use_container_width=True)
                            if link:
                                st.link_button("📲 İndir", link, use_container_width=True)

    # ── Yeni Kullanıcı Girişi ──
    if chat_prompt := st.chat_input("Archi'ye bir şeyler sor..."):
        
        chat_temp_path = None
        if chat_uploaded_file:
            import hashlib
            chat_file_hash = hashlib.md5(chat_uploaded_file.getbuffer()).hexdigest()
            chat_temp_path = str(Path("./data/temp") / f"chat_{chat_file_hash}.jpg")
            Path(chat_temp_path).parent.mkdir(parents=True, exist_ok=True)
            with open(chat_temp_path, "wb") as f:
                f.write(chat_uploaded_file.getbuffer())
        
        user_msg = {"role": "user", "content": chat_prompt, "results": [], "image": chat_temp_path}
        st.session_state.chat_history.append(user_msg)
        
        with st.chat_message("user", avatar="👤"):
            st.markdown(chat_prompt)
            if chat_temp_path:
                st.image(chat_temp_path, width=250)
            
        with st.chat_message("assistant", avatar="🤖"):
            archi_response = chat_with_designer_v2(
                chat_prompt,
                st.session_state.chat_history,
                image_path=chat_temp_path,
            )
            reply_text = archi_response.get("reply_text", "")
            plan = archi_response.get("plan", {})
            search_queries = archi_response.get("search_queries", [])

            found_results = []
            result_explanation = ""
            
            if search_queries:
                with st.status("Arşivde tarıyorum...", expanded=False) as status:
                    seen_ids = set()
                    per_query = max(6, min(24, int(n_results / max(1, min(len(search_queries), 4)))))
                    for sq in search_queries[:6]:
                        batch_results, _ = search_by_text(sq, store, n_results=per_query, use_ai_expansion=False)
                        for item in batch_results:
                            item_id = item.get("id") or (item.get("metadata") or {}).get("image_path")
                            if item_id not in seen_ids:
                                seen_ids.add(item_id)
                                found_results.append(item)
                    found_results = annotate_results(found_results, plan)[:n_results]
                    result_explanation = explain_results(plan, found_results) if found_results else ""
                    update_archi_memory(chat_prompt, plan, found_results)
                    status.update(label=f"{len(found_results)} model bulundu!", state="complete", expanded=False)
            else:
                update_archi_memory(chat_prompt, plan, [])
            
            st.markdown(reply_text)
            
            if found_results:
                st.markdown(f'<div style="margin:12px 0 8px; padding:8px 12px; background:rgba(102,126,234,0.08); border-radius:10px; border-left:3px solid #667eea; font-size:0.82rem; color:#8eaaff; font-weight:600;">✨ {len(found_results)} model bulundu</div>', unsafe_allow_html=True)
                cols = st.columns(3)
                for idx, res in enumerate(found_results[:9]):
                    with cols[idx % 3]:
                        meta = res.get("metadata", {})
                        img_path = meta.get("image_path", "")
                        link = meta.get("deep_link", "")
                        reasons = res.get("archi_reasons", [])
                        if img_path and Path(img_path).exists():
                            st.image(img_path, use_container_width=True)
                            if reasons:
                                st.caption("; ".join(reasons[:2]))
                            if link:
                                st.link_button("📲 İndir", link, use_container_width=True)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": reply_text + ("\n\n" + result_explanation if result_explanation else ""),
                "results": found_results,
                "archi_plan": plan,
            })
            
            st.session_state.chat_uploader_key += 1
            st.rerun()



