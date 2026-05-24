"""
🤖 Archi Telegram Bot — Cebindeki 3D Model Arama Motoru

527.000+ modelli arşivde yapay zeka (SigLIP 2) ile arama yapan Telegram botu.
Metin veya fotoğraf gönder → en benzer modelleri al → deep link ile orijinale git.
"""
import os
import sys
import logging
import tempfile
from pathlib import Path

from telegram import (
    Update,
    InputMediaPhoto,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    BotCommand,
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)
from telegram.constants import ParseMode, ChatAction

# Proje kökünü Python path'e ekle
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_settings, MetadataSchema
from indexer.chroma_store import ChromaStore
from search.text_search import search_by_text
from search.image_search import search_by_image

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════
# Sabitler
# ══════════════════════════════════════════════════

MAX_RESULTS = 5          # Telegram'da gösterilecek maksimum sonuç
MIN_SCORE = 0.05         # Minimum benzerlik eşiği
MIN_AESTHETIC = 0.0      # Minimum estetik skor (0 = filtre yok)
TEMP_DIR = PROJECT_ROOT / "data" / "temp" / "bot_queries"

# ══════════════════════════════════════════════════
# ChromaDB Bağlantısı (Singleton)
# ══════════════════════════════════════════════════

_store = None

def get_store() -> ChromaStore:
    """ChromaDB bağlantısını lazy-load ile oluşturur."""
    global _store
    if _store is None:
        settings = get_settings()
        _store = ChromaStore(
            db_path=settings.chroma_db_path,
            collection_name=settings.get_collection_name(),
        )
        count = _store.get_count()
        logger.info(f"📦 ChromaDB bağlantısı kuruldu: {count:,} kayıt")
    return _store


# ══════════════════════════════════════════════════
# Yardımcı Fonksiyonlar
# ══════════════════════════════════════════════════

def format_result_caption(result: dict, rank: int) -> str:
    """Tek bir arama sonucu için güzel bir caption oluşturur."""
    meta = result.get("metadata", {})
    score = result.get("score", 0)

    # Estetik skor
    aes = meta.get(MetadataSchema.AESTHETIC_SCORE, 0)
    try:
        aes = float(aes) if aes else 0
    except (ValueError, TypeError):
        aes = 0

    if aes >= 8:
        aes_icon = "🏆"
    elif aes >= 6:
        aes_icon = "🌟"
    elif aes >= 4:
        aes_icon = "👍"
    else:
        aes_icon = "📷"

    # Kanal bilgisi
    channel = meta.get(MetadataSchema.CHANNEL_TITLE, "")
    if not channel:
        channel = meta.get(MetadataSchema.CHANNEL_USERNAME, "Bilinmeyen")

    # SigLIP etiketleri
    tags_parts = []
    for key in ["clip_furniture_type", "clip_style", "clip_material"]:
        val = meta.get(key, "")
        if val and val != "unknown":
            tags_parts.append(str(val))

    tags_str = " · ".join(tags_parts[:3]) if tags_parts else ""

    lines = [f"{aes_icon} #{rank}  ·  Benzerlik: {score:.0%}"]
    if aes > 0:
        lines.append(f"✨ Estetik: {aes:.1f}/10")
    if tags_str:
        lines.append(f"🏷 {tags_str}")
    lines.append(f"📢 {channel}")

    return "\n".join(lines)


def get_result_keyboard(result: dict) -> InlineKeyboardMarkup:
    """Bir sonuç için 'Telegram'da Aç' butonunu oluşturur."""
    meta = result.get("metadata", {})
    deep_link = meta.get(MetadataSchema.DEEP_LINK, "")

    buttons = []
    if deep_link:
        buttons.append([InlineKeyboardButton("📬 Telegram'da Aç", url=deep_link)])

    return InlineKeyboardMarkup(buttons) if buttons else None


def get_image_path(result: dict) -> str | None:
    """Sonuçtaki görsel dosya yolunu döndürür, yoksa None."""
    meta = result.get("metadata", {})
    image_path = meta.get(MetadataSchema.IMAGE_PATH, "")

    if not image_path:
        return None

    path = Path(image_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / image_path

    return str(path) if path.exists() else None


def filter_by_aesthetic(results: list, min_aes: float) -> list:
    """Estetik skora göre filtreler."""
    if min_aes <= 0:
        return results

    filtered = []
    for r in results:
        meta = r.get("metadata", {})
        try:
            aes = float(meta.get(MetadataSchema.AESTHETIC_SCORE, 0) or 0)
        except (ValueError, TypeError):
            aes = 0
        if aes >= min_aes:
            filtered.append(r)
    return filtered


# ══════════════════════════════════════════════════
# Bot Komutları
# ══════════════════════════════════════════════════

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Hoş geldin mesajı."""
    store = get_store()
    count = store.get_count()

    welcome = (
        "🔍 *Archi — 3D Model Arama Motoru*\n\n"
        f"Merhaba\\! {count:,} modelli devasa arşivde yapay zeka ile arama yapabilirsin\\.\n\n"
        "💬 *Nasıl Kullanılır?*\n"
        "• Bir şey yaz → _modern ahşap sehpa_\n"
        "• Fotoğraf gönder → benzer modelleri bulur\n"
        "• Bir kanaldan fotoğraf forward et → benzerlerini getirir\n\n"
        "⚙️ *Komutlar*\n"
        "• /ara \\<sorgu\\> — Metin ile ara\n"
        "• /stats — Arşiv istatistikleri\n"
        "• /help — Detaylı yardım"
    )

    await update.message.reply_text(welcome, parse_mode=ParseMode.MARKDOWN_V2)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Detaylı yardım mesajı."""
    help_text = (
        "📖 *Archi Kullanım Kılavuzu*\n\n"
        "*1\\. Metin Araması*\n"
        "Doğal dilde yaz, yapay zeka anlasın:\n"
        "`modern koltuk`\n"
        "`mermer masa altın detaylı`\n"
        "`İskandinav tarzı ahşap sandalye`\n\n"
        "*2\\. Görsel Araması*\n"
        "Pinterest, Google veya herhangi bir yerden bulduğun\n"
        "referans fotoğrafı bota gönder\\. En benzer modelleri getirir\\.\n\n"
        "*3\\. Forward Araması*\n"
        "Telegram kanallarında gördüğün bir modeli bu bota\n"
        "forward et\\. Arşivdeki benzerlerini bulur\\.\n\n"
        "*4\\. Komutlar*\n"
        "• /ara \\<sorgu\\> — Metin ile ara\n"
        "• /stats — Arşiv durumu\n"
        "• /elite — Sadece elit kalite modeller"
    )

    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN_V2)


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Arşiv istatistiklerini gösterir."""
    store = get_store()
    settings = get_settings()

    count = store.get_count()
    model = settings.clip_model_name
    collection = settings.get_collection_name()

    images_dir = settings.get_images_path()
    img_count = sum(1 for _ in images_dir.glob("*.jpg")) if images_dir.exists() else 0

    stats = (
        "📊 *Archi Arşiv İstatistikleri*\n\n"
        f"🗄 İndeksli Model: *{count:,}*\n"
        f"🖼 İndirilen Görsel: *{img_count:,}*\n"
        f"🧠 AI Model: `{model}`\n"
        f"📦 Koleksiyon: `{collection}`"
    )

    await update.message.reply_text(stats, parse_mode=ParseMode.MARKDOWN_V2)


async def cmd_search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/ara <sorgu> komutuyla metin araması."""
    query = " ".join(context.args) if context.args else ""
    if not query.strip():
        await update.message.reply_text("❌ Kullanım: /ara modern koltuk")
        return

    await _handle_text_search(update, query)


async def cmd_elite(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/elite — sadece elit kalite modelleri arar."""
    query = " ".join(context.args) if context.args else ""
    if not query.strip():
        await update.message.reply_text("❌ Kullanım: /elite modern koltuk")
        return

    await _handle_text_search(update, query, min_aesthetic=7.0)


# ══════════════════════════════════════════════════
# Mesaj İşleyiciler
# ══════════════════════════════════════════════════

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Düz metin mesajlarını otomatik arama olarak işler."""
    query = update.message.text.strip()
    if not query or query.startswith("/"):
        return

    await _handle_text_search(update, query)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Gönderilen veya forward edilen fotoğraflarla görsel araması yapar."""
    await update.message.chat.send_action(action=ChatAction.TYPING)

    # En yüksek çözünürlüklü fotoğrafı al
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)

    # Geçici dosyaya indir
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = TEMP_DIR / f"query_{update.message.message_id}.jpg"
    await file.download_to_drive(str(temp_path))

    try:
        await update.message.reply_text("🔍 Benzer modeller aranıyor...")

        store = get_store()
        results, analysis = search_by_image(
            image_path=str(temp_path),
            store=store,
            n_results=MAX_RESULTS * 3,  # Fazla getir, filtrele
            min_score=MIN_SCORE,
            use_ai_expansion=False,  # Bot'ta hız önemli, LLM analizi atla
        )

        # Estetik filtre
        results = filter_by_aesthetic(results, MIN_AESTHETIC)
        results = results[:MAX_RESULTS]

        if not results:
            await update.message.reply_text(
                "😔 Maalesef benzer model bulunamadı. "
                "Farklı bir açıdan çekilmiş fotoğraf deneyin."
            )
            return

        await _send_results(update, results, f"🖼 Görsel araması")

    finally:
        # Geçici dosyayı temizle
        if temp_path.exists():
            temp_path.unlink()


# ══════════════════════════════════════════════════
# Ortak Arama & Gönderim Mantığı
# ══════════════════════════════════════════════════

async def _handle_text_search(
    update: Update,
    query: str,
    min_aesthetic: float = MIN_AESTHETIC,
):
    """Metin araması yapar ve sonuçları gönderir."""
    await update.message.chat.send_action(action=ChatAction.TYPING)
    await update.message.reply_text(f"🔍 Aranıyor: _{query}_", parse_mode=ParseMode.MARKDOWN)

    store = get_store()
    results, expanded = search_by_text(
        query=query,
        store=store,
        n_results=MAX_RESULTS * 3,
        min_score=MIN_SCORE,
    )

    # Estetik filtre
    results = filter_by_aesthetic(results, min_aesthetic)
    results = results[:MAX_RESULTS]

    if not results:
        await update.message.reply_text(
            "😔 Sonuç bulunamadı. Farklı kelimelerle tekrar deneyin.\n"
            "💡 İpucu: İngilizce aramalar daha iyi sonuç verir."
        )
        return

    header = f"🔍 \"{query}\""
    if min_aesthetic > 0:
        header += f" (Elit: {min_aesthetic}+)"
    await _send_results(update, results, header)


async def _send_results(update: Update, results: list, header: str):
    """Arama sonuçlarını fotoğraf albümü + butonlar olarak gönderir."""

    # Sonuçları teker teker gönder (her biri fotoğraf + caption + buton)
    await update.message.reply_text(
        f"🎯 *{len(results)} sonuç bulundu* — {header}",
        parse_mode=ParseMode.MARKDOWN,
    )

    for rank, result in enumerate(results, 1):
        image_path = get_image_path(result)
        caption = format_result_caption(result, rank)
        keyboard = get_result_keyboard(result)

        try:
            if image_path:
                with open(image_path, "rb") as photo_file:
                    await update.message.reply_photo(
                        photo=photo_file,
                        caption=caption,
                        reply_markup=keyboard,
                    )
            else:
                # Görsel dosya yoksa sadece metin gönder
                meta = result.get("metadata", {})
                deep_link = meta.get(MetadataSchema.DEEP_LINK, "")
                text = caption
                if deep_link:
                    text += f"\n\n🔗 {deep_link}"
                await update.message.reply_text(text)

        except Exception as e:
            logger.warning(f"Sonuç #{rank} gönderilemedi: {e}")
            continue


# ══════════════════════════════════════════════════
# Bot Başlatma
# ══════════════════════════════════════════════════

def create_bot() -> Application:
    """Bot uygulamasını oluşturur ve handler'ları kayıt eder."""
    settings = get_settings()
    token = settings.telegram_bot_token

    if not token:
        raise ValueError(
            "❌ TELEGRAM_BOT_TOKEN bulunamadı!\n"
            ".env dosyasına ekleyin: TELEGRAM_BOT_TOKEN=<token>"
        )

    app = Application.builder().token(token).build()

    # Komutlar
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("ara", cmd_search))
    app.add_handler(CommandHandler("elite", cmd_elite))

    # Mesaj handler'ları (sıra önemli!)
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND,
        handle_text_message,
    ))

    return app


async def post_init(app: Application):
    """Bot başladığında komut menüsünü ayarlar."""
    commands = [
        BotCommand("start", "🚀 Başlat"),
        BotCommand("ara", "🔍 Metin ile ara"),
        BotCommand("elite", "🏆 Sadece elit modeller"),
        BotCommand("stats", "📊 Arşiv istatistikleri"),
        BotCommand("help", "📖 Yardım"),
    ]
    await app.bot.set_my_commands(commands)
    logger.info("✅ Bot komut menüsü ayarlandı")


def run():
    """Botu başlatır (polling modu)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("🤖 Archi Telegram Bot başlatılıyor...")

    # ChromaDB'yi önceden yükle
    store = get_store()
    logger.info(f"📦 Arşiv hazır: {store.get_count():,} model")

    # Bot oluştur
    app = create_bot()
    app.post_init = post_init

    logger.info("✅ Bot aktif! Telegram'da mesaj bekleniyor...")
    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,  # Eski mesajları atla
    )


if __name__ == "__main__":
    run()
