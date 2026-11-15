"""1.30: salidas sintéticas del modo de pruebas (CINESTORIA_FAKE=1): la forma exacta de las de verdad (PNG del tamaño
pedido, WAV con su frecuencia, MP4 de la duración pedida), sin torch ni GPU. Solo biblioteca estándar (y ffmpeg para
el MP4, si está; si no, unos bytes con la cabecera de un MP4)."""
from __future__ import annotations

import base64
import io
import math
import os
import shutil
import struct
import subprocess
import tempfile
import wave
import zlib


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def png(width: int, height: int, seed: int = 0) -> bytes:
    """Un PNG RGB de width×height: un degradado con un color que depende de la semilla."""
    w, h = max(1, int(width)), max(1, int(height))
    r0, g0, b0 = (seed * 53) % 200 + 30, (seed * 97) % 200 + 30, (seed * 31) % 200 + 30
    rows = []
    for y in range(h):
        k = y / max(1, h - 1)
        px = bytes((int(r0 * (1 - k * 0.5)), int(g0 * (0.6 + k * 0.4)), int(b0 * (1 - k * 0.3))))
        rows.append(b"\x00" + px * w)

    def chunk(t: bytes, d: bytes) -> bytes:
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(rows), 6)) + chunk(b"IEND", b""))


def png_size(data: bytes) -> tuple[int, int]:
    """(ancho, alto) de un PNG (para comprobar la forma)."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("no es un PNG")
    return struct.unpack(">II", data[16:24])


def wav(seconds: float, sr: int = 24000, freq: float = 220.0, seed: int = 0) -> bytes:
    """Un WAV mono de 16 bits: un tono suave (con la semilla, otra nota)."""
    n = max(1, int(round(float(seconds) * sr)))
    f = freq * (1 + (seed % 7) / 12)
    frames = b"".join(struct.pack("<h", int(1800 * math.sin(2 * math.pi * f * i / sr))) for i in range(n))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(frames)
    return buf.getvalue()


def wav_info(data: bytes) -> tuple[int, float]:
    """(frecuencia, segundos) de un WAV."""
    with wave.open(io.BytesIO(data), "rb") as w:
        return w.getframerate(), w.getnframes() / w.getframerate()


def words(text: str, seconds: float) -> list[dict]:
    """Los tiempos de cada palabra, repartidos por su largo."""
    ws = [w for w in str(text or "").split() if w]
    if not ws:
        return []
    total = sum(len(w) + 1 for w in ws)
    t, out = 0.0, []
    for w in ws:
        d = float(seconds) * (len(w) + 1) / total
        out.append({"text": w, "start": round(t, 3), "end": round(t + d * 0.92, 3)})
        t += d
    return out


def mp4(seconds: float, width: int, height: int, fps: int = 24) -> bytes:
    """Un MP4 de esa duración y tamaño (con ffmpeg); sin ffmpeg, una cabecera «ftyp» válida para la forma."""
    ff = os.environ.get("NARRA_FFMPEG") or shutil.which("ffmpeg")
    w, h = max(2, int(width) // 2 * 2), max(2, int(height) // 2 * 2)
    if ff:
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "c.mp4")
            try:
                subprocess.run([ff, "-v", "error", "-y", "-f", "lavfi", "-i",
                                f"color=c=0x404850:s={w}x{h}:d={max(0.2, float(seconds)):.3f}:r={int(fps or 24)}",
                                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", out],
                               check=True, timeout=120, capture_output=True)
                with open(out, "rb") as f:
                    return f.read()
            except (OSError, subprocess.SubprocessError):
                pass
    return struct.pack(">I", 24) + b"ftypisom" + struct.pack(">I", 512) + b"isomiso2"
