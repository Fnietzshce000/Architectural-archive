"""
Telegram Deep Link Oluşturucu.
Public ve private kanallar için doğru URL formatını üretir.
"""


def generate_deep_link(
    channel_id: int,
    channel_username: str | None,
    message_id: int,
) -> str:
    """
    Telegram deep link oluşturur.

    Public kanal:  https://t.me/kanaladi/mesaj_id
    Private kanal: https://t.me/c/saf_id/mesaj_id

    Args:
        channel_id: Kanalın Telegram ID'si
        channel_username: Public kanallar için @username (private ise None)
        message_id: Mesaj numarası

    Returns:
        Tıklanabilir Telegram deep link URL'si
    """
    if channel_username:
        username = str(channel_username).lstrip("@")
        return f"https://t.me/{username}/{message_id}"
    else:
        # Private kanal: -100 önekini kırp
        raw_id = str(abs(channel_id))
        if raw_id.startswith("100"):
            raw_id = raw_id[3:]
        return f"https://t.me/c/{raw_id}/{message_id}"


def format_file_size(size_bytes: int) -> str:
    """Dosya boyutunu okunabilir formata çevirir."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / 1024**2:.1f} MB"
    else:
        return f"{size_bytes / 1024**3:.1f} GB"
