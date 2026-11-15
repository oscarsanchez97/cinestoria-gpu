"""1.30 (voz): la narración de Cinestoria en la GPU del cliente («Tu GPU»).

POST /tts    la voz de un tramo (una o varias frases seguidas): cada frase se sintetiza por separado con Chatterbox
             Multilingual V3 (Resemble AI, licencia MIT; en español de España, de Latinoamérica y en portugués de Brasil y
             de Portugal, con su paquete de idioma, también MIT), con la voz de referencia de una voz de serie de
             Cinestoria (voces.json: diseñadas con Qwen3-TTS-VoiceDesign, Apache-2.0, a partir de una descripción; nunca el
             clon de nadie) o la que sube el creador (la suya). Cada frase pasa un CONTROL antes de devolverse: se vuelve a
             transcribir con Whisper large-v3-turbo (MIT) y se mira que diga lo que tiene que decir (WER), que dure lo que
             es razonable, que no tenga silencios raros, que no sature, que no tenga golpes ni ruidos y que suene a su
             nivel. Si no lo pasa, se rehace con otra semilla (hasta `retries` veces) y se devuelve la mejor, con su nota:
             el cliente decide (rescate premium con ElevenLabs si lo tiene). Los tiempos de cada palabra salen de una
             alineación forzada CTC (wav2vec2 XLSR-53 de cada idioma, Apache-2.0), como los de ElevenLabs.
POST /align  los tiempos de cada palabra de un audio y su texto (la misma alineación).

Contrato 1.30 (contrato130.md) más campos opcionales (los de siempre siguen valiendo):
  /tts  entrada: text, lang, voice, voice_ref, speed, style, seed (contrato) · chunks (las frases, en orden, que juntas
        son `text`), alts (otra forma de cada frase para el control: la escrita cuando se manda la fonética), styles (la
        expresividad de cada frase, 0-1), pauses (el silencio entre frases), temperature, variant, retries, check
        salida: audio, sr, words, seconds (segundos de GPU), model (contrato) · duration (segundos de audio), voice,
        chunks [{text, start, end, ok, wer, why, tries}], voice_ref (la referencia usada, si no venía en la petición)
Modo de prueba (CINESTORIA_FAKE=1): sin torch ni modelos, un audio sintético con la forma de verdad (palabras con su
tiempo exacto, el control y los reintentos de verdad) para probar el contrato en una CPU. Los modelos se cargan solos
la primera vez (importaciones dentro de las funciones) y la memoria de la GPU se pide con vram.ensure.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import unicodedata
import wave
import zlib
from pathlib import Path
from typing import Any

import numpy as np

VERSION = "1.30"
SR = 24000                 # lo que devuelve /tts (lo mismo que la voz de Cinestoria)
ASR_SR = 16000             # Whisper y wav2vec2
MODEL = "chatterbox-v3"
HERE = Path(__file__).resolve().parent
VOICES_FILE = HERE / "voces.json"
LANGS = ("es", "en", "pt", "fr", "de", "it")
MAX_CHARS = 300            # lo más largo que se le da a Chatterbox de una vez (sus demos cortan en 300)
LEAD, TAIL = 0.12, 0.25    # el silencio al principio y al final de un tramo
PAUSE = 0.32               # entre frases, si el cliente no dice otra cosa
SUB_PAUSE = 0.16           # entre los trozos de una frase muy larga (en su coma)
TARGET_DB = -20.0          # el nivel de la voz donde suena (cada frase igual: como sale de ElevenLabs)
PEAK = 0.95

# los paquetes de un solo idioma de Chatterbox V3 (MIT): https://github.com/resemble-ai/chatterbox
PACK_REPOS = {"es-es": "ResembleAI/Chatterbox-Multilingual-es-es", "es-mx": "ResembleAI/Chatterbox-Multilingual-es-mx-latam",
              "pt-br": "ResembleAI/Chatterbox-Multilingual-pt-br", "pt-pt": "ResembleAI/Chatterbox-Multilingual-pt-pt"}
ALIGNERS = {"es": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish", "en": "jonatasgrosman/wav2vec2-large-xlsr-53-english",
            "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese", "fr": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
            "de": "jonatasgrosman/wav2vec2-large-xlsr-53-german", "it": "jonatasgrosman/wav2vec2-large-xlsr-53-italian"}
ASR_REPO = "openai/whisper-large-v3-turbo"
DESIGN_REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
WHISPER_LANG = {"es": "spanish", "en": "english", "pt": "portuguese", "fr": "french", "de": "german", "it": "italian"}

# Los pesos que usa esta parte (models.py de núcleo los descarga y los guarda; licencias comprobadas en su ficha de
# Hugging Face el 15-11-2026, ver NOTAS.md). "gb": lo que ocupan en disco, más o menos.
MODELS: list[dict[str, Any]] = [
    {"name": "chatterbox-v3", "repo": "ResembleAI/chatterbox", "licence": "MIT", "gb": 3.2,
     "use": "la voz (23 idiomas)"},
    *[{"name": f"chatterbox-{p}", "repo": r, "licence": "MIT", "gb": 2.2, "use": f"la voz ({p})"}
      for p, r in PACK_REPOS.items()],
    {"name": "whisper-large-v3-turbo", "repo": ASR_REPO, "licence": "MIT", "gb": 1.6, "use": "el control de cada frase"},
    *[{"name": f"wav2vec2-{k}", "repo": r, "licence": "Apache-2.0", "gb": 1.3, "use": f"los tiempos de cada palabra ({k})"}
      for k, r in ALIGNERS.items()],
    {"name": "qwen3-tts-voicedesign", "repo": DESIGN_REPO, "licence": "Apache-2.0", "gb": 4.5, "optional": True,
     "use": "diseñar una voz de serie la primera vez (después, su referencia se guarda)"},
]
# lo que necesita en la GPU (contrato: un A40/A6000 de 48 GB tiene que caber con lo de cada etapa)
VRAM_GB = {"tts": 10.0, "diseno": 7.0}

# el control de cada frase
WER_MAX = 0.12             # palabras mal dichas, de más o de menos (como mucho 1 en las frases de menos de 9)
DUR_RANGE = (0.55, 1.8)    # lo que dura la voz frente a lo que se espera por sus sílabas
SYL_PER_S = {"es": 5.2, "it": 5.2, "pt": 5.0, "fr": 5.0, "de": 4.2, "en": 4.0}   # a ritmo de narración
GAP_MAX = 1.0              # un silencio dentro de una frase de más de esto: se ha atascado
CLIP_MAX = 0.002           # muestras saturadas
BURST_DB = 10.0            # un golpe: un trozo de 20 ms tanto por encima de lo más fuerte de la voz (su percentil 90)
QUIET_DB = -45.0           # una frase casi muda

_LOCK = threading.Lock()   # un modelo de voz no se usa desde dos hilos a la vez (Chatterbox guarda la voz en sí)
_ASR_LOCK = threading.Lock()
_DESIGN_LOCK = threading.Lock()
_REF_LOCK = threading.Lock()   # (una voz de serie se diseña una sola vez aunque lleguen dos tramos a la vez)
_M: dict[str, Any] = {}    # los modelos cargados
NOTES: list[str] = []      # lo que no salió como se esperaba al cargar (para /health y las notas)


class BadRequest(ValueError):
    pass


def fake() -> bool:
    return os.environ.get("CINESTORIA_FAKE") == "1"


def cache_dir() -> Path:
    d = Path(os.environ.get("CINESTORIA_CACHE") or "/workspace/cinestoria") / "voces"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        import tempfile
        d = Path(tempfile.gettempdir()) / "cinestoria-voces"
        d.mkdir(parents=True, exist_ok=True)
    return d


def voices() -> list[dict[str, Any]]:
    try:
        return list(json.loads(VOICES_FILE.read_text(encoding="utf-8")).get("voces") or [])
    except (OSError, ValueError):
        return []


def voice_spec(voice_id: str | None) -> dict[str, Any] | None:
    return next((v for v in voices() if v.get("id") == voice_id), None) if voice_id else None


# ---------- audio ----------

def wav_bytes(x: np.ndarray, sr: int = SR) -> bytes:
    pcm = (np.clip(np.nan_to_num(np.asarray(x, np.float32)), -1, 1) * 32767).astype("<i2")
    bio = io.BytesIO()
    with wave.open(bio, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())
    return bio.getvalue()


def b64wav(x: np.ndarray, sr: int = SR) -> str:
    return base64.b64encode(wav_bytes(x, sr)).decode()


def read_wav(data: bytes) -> tuple[np.ndarray, int]:
    """Un WAV (PCM de 16 bits, o lo que entienda soundfile) a mono float32."""
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
            raw = w.readframes(n)
        if sw != 2:
            raise wave.Error("no es PCM de 16 bits")
        x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
        if ch > 1:
            x = x.reshape(-1, ch).mean(axis=1)
        return x, sr
    except (wave.Error, EOFError, ValueError):
        import soundfile as sf
        x, sr = sf.read(io.BytesIO(data), dtype="float32")
        return (x.mean(axis=1) if x.ndim > 1 else x), int(sr)


def from_b64(s: str) -> tuple[np.ndarray, int]:
    try:
        return read_wav(base64.b64decode(s))
    except Exception as e:  # noqa: BLE001
        raise BadRequest(f"audio no válido: {e}") from e


def resample(x: np.ndarray, sr: int, to: int) -> np.ndarray:
    x = np.asarray(x, np.float32)
    if sr == to or len(x) == 0:
        return x
    try:   # (en la GPU, el remuestreo bueno; aquí, interpolación)
        if not fake():
            import torch
            import torchaudio
            return torchaudio.functional.resample(torch.from_numpy(x), sr, to).numpy().astype(np.float32)
    except Exception:  # noqa: BLE001
        pass
    n = int(round(len(x) * to / sr))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def frames_db(x: np.ndarray, sr: int, ms: float = 20.0) -> np.ndarray:
    hop = max(1, int(sr * ms / 1000))
    n = len(x) // hop
    if n == 0:
        return np.full(1, -120.0)
    e = (np.asarray(x[: n * hop], np.float64).reshape(n, hop) ** 2).mean(axis=1)
    return 10 * np.log10(e + 1e-12)


def voiced_db(x: np.ndarray, sr: int) -> float:
    """El nivel de la voz donde suena (los trozos a menos de 35 dB de lo más fuerte), en dBFS."""
    f = frames_db(x, sr)
    top = float(f.max())
    if top < -100:
        return -120.0
    v = f[f > top - 35]
    return float(10 * np.log10(np.mean(10 ** (v / 10)) + 1e-12))


def normalize(x: np.ndarray, sr: int) -> np.ndarray:
    """Cada frase al mismo nivel (TARGET_DB donde suena) y sin pasar de PEAK (se baja entera: nunca se distorsiona)."""
    x = np.nan_to_num(np.asarray(x, np.float32))
    lv = voiced_db(x, sr)
    if lv < -100:
        return x
    y = x * np.float32(10 ** ((TARGET_DB - lv) / 20))
    peak = float(np.abs(y).max())
    if peak > PEAK:
        y *= np.float32(PEAK / peak)
    return y.astype(np.float32)


def voiced_span(x: np.ndarray, sr: int) -> tuple[float, float]:
    f = frames_db(x, sr)
    top = float(f.max())
    on = np.where(f > max(top - 35, -70))[0]
    if len(on) == 0:
        return 0.0, 0.0
    return on[0] * 0.02, (on[-1] + 1) * 0.02


def trim(x: np.ndarray, sr: int, keep: float = 0.03) -> np.ndarray:
    a, b = voiced_span(x, sr)
    if b <= a:
        return x
    return x[max(0, int((a - keep) * sr)):min(len(x), int((b + keep * 2) * sr))]


def internal_gaps(x: np.ndarray, sr: int) -> list[float]:
    """Los silencios dentro de la voz (entre su primer y su último sonido), en segundos."""
    f = frames_db(x, sr)
    top = float(f.max())
    on = f > max(top - 35, -70)
    idx = np.where(on)[0]
    if len(idx) < 2:
        return []
    out, run = [], 0
    for v in on[idx[0]:idx[-1] + 1]:
        if v:
            if run:
                out.append(run * 0.02)
            run = 0
        else:
            run += 1
    return out


# ---------- texto ----------

_VOWELS = re.compile(r"[aeiouyáéíóúàèìòùâêîôûäëïöüãõœæ]+", re.I)
_EDGE = re.compile(r"^([^\w]*)(.*?)([^\w]*)$", re.S)
_PCT = {"es": "por ciento", "en": "percent", "pt": "por cento", "fr": "pour cent", "de": "Prozent", "it": "per cento"}


def deaccent(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _number(core: str, lang: str) -> list[str] | None:
    """«1912», «13.000», «5,5», «40 %»… dichos con palabras (num2words, si está). None si no es un número o no se puede."""
    m = re.fullmatch(r"(\d[\d.,\s]*)(%?)", core)
    if not m:
        return None
    try:
        from num2words import num2words
    except ImportError:
        return None
    s = m.group(1).replace(" ", "")
    dec = "." if lang == "en" else ","
    th = "," if lang == "en" else "."
    try:
        if re.fullmatch(r"\d{1,3}(" + re.escape(th) + r"\d{3})+", s):
            val: float | int = int(s.replace(th, ""))
        elif s.count(dec) == 1 and re.fullmatch(r"\d+" + re.escape(dec) + r"\d+", s):
            val = float(s.replace(dec, "."))
        elif s.isdigit():
            val = int(s)
        else:
            return None
        kw = {"to": "year"} if lang == "en" and isinstance(val, int) and 1100 <= val <= 2099 and len(s) == 4 else {}
        words = str(num2words(val, lang=lang, **kw)).replace("-", " ").split()
    except Exception:  # noqa: BLE001
        return None
    return words + (_PCT.get(lang, "%").split() if m.group(2) else [])


def speakable(tok: str, lang: str) -> tuple[str, list[str]]:
    """Cómo se le da una palabra a la voz (los números, con letras) y las palabras que hay que encontrar en el audio."""
    m = _EDGE.match(tok)
    lead, core, trail = (m.group(1), m.group(2), m.group(3)) if m else ("", tok, "")
    if not core:
        return tok, []
    nw = _number(core, lang) if any(c.isdigit() for c in core) else None
    if nw:
        return lead + " ".join(nw) + trail, nw
    return tok, [core]


def norm_words(text: str, lang: str) -> list[str]:
    """Las palabras de un texto para compararlas (sin mayúsculas, acentos ni signos; los números, con letras)."""
    out: list[str] = []
    for tok in str(text or "").split():
        _, ws = speakable(tok, lang)
        for w in ws:
            w = re.sub(r"[^\w]+", " ", deaccent(w).lower()).replace("_", " ")
            out += w.split()
    return out


def wer(ref: list[str], hyp: list[str]) -> float:
    """Proporción de palabras mal (cambiadas, de más o de menos) frente a las que tenía que decir."""
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / len(ref)


def syllables(words: list[str]) -> int:
    return sum(max(1, len(_VOWELS.findall(w))) for w in words)


def chunk_ranges(toks: list[str], chunks: Any = None) -> list[tuple[int, int]]:
    """Las frases del tramo como rangos de palabras. Si el cliente las manda y juntas son el texto, esas; si no, partido
    por los puntos."""
    if isinstance(chunks, list) and chunks and all(isinstance(c, str) for c in chunks):
        parts = [c.split() for c in chunks]
        if [w for p in parts for w in p] == toks and all(parts):
            out, a = [], 0
            for p in parts:
                out.append((a, a + len(p)))
                a += len(p)
            return out
    out, a = [], 0
    for i, w in enumerate(toks):
        if re.search(r"[.!?…][»”\"')\]]*$", w) or i == len(toks) - 1:
            out.append((a, i + 1))
            a = i + 1
    return out or [(0, len(toks))]


def sub_ranges(toks: list[str], a: int, b: int, limit: int = MAX_CHARS) -> list[tuple[int, int]]:
    """Una frase de más de `limit` caracteres, en trozos (en sus comas, o donde toque) para Chatterbox."""
    if len(" ".join(toks[a:b])) <= limit:
        return [(a, b)]
    out, s, n = [], a, 0
    for i in range(a, b):
        n += len(toks[i]) + 1
        last = i == b - 1
        if not last and (n >= limit * 0.85 or (n >= limit * 0.4 and re.search(r"[,;:—–]$", toks[i]))):
            out.append((s, i + 1))
            s, n = i + 1, 0
    if s < b:
        out.append((s, b))
    return out


# ---------- alineación forzada (CTC) ----------

def ctc_align(logp: np.ndarray, targets: list[int], blank: int = 0) -> list[tuple[int, int]] | None:
    """Viterbi de CTC: dónde suena cada letra (primer y último fotograma de su tramo). `logp`: [T, V] log-probabilidades.
    None si no cabe (más letras que fotogramas) o no hay camino."""
    T = int(logp.shape[0])
    L = len(targets)
    if L == 0:
        return []
    S = 2 * L + 1
    ext = np.full(S, blank, dtype=np.int64)
    ext[1::2] = targets
    skip = np.zeros(S, bool)
    if L > 1:
        skip[3::2] = ext[3::2] != ext[1:-2:2]
    neg = -1e30
    dp = np.full(S, neg)
    dp[0] = logp[0, blank]
    dp[1] = logp[0, ext[1]]
    back = np.zeros((T, S), np.int8)
    for t in range(1, T):
        b1 = np.full(S, neg)
        b1[1:] = dp[:-1]
        b2 = np.full(S, neg)
        b2[2:] = np.where(skip[2:], dp[:-2], neg)
        st = np.stack([dp, b1, b2])
        arg = st.argmax(axis=0)
        back[t] = arg
        dp = st[arg, np.arange(S)] + logp[t, ext]
    end = S - 1 if dp[S - 1] >= dp[S - 2] else S - 2
    if dp[end] < neg / 2:
        return None
    states = np.empty(T, np.int64)
    s = end
    for t in range(T - 1, -1, -1):
        states[t] = s
        if t:
            s -= int(back[t, s])
    spans: list[list[int]] = [[-1, -1] for _ in range(L)]
    for t, s in enumerate(states):
        if s % 2:
            k = int(s) // 2
            if spans[k][0] < 0:
                spans[k][0] = t
            spans[k][1] = t
    if any(a < 0 for a, _ in spans):
        return None
    return [(a, b) for a, b in spans]


def char_ids(word: str, vocab: dict[str, int]) -> list[int]:
    out = []
    for c in word:
        for cand in (c, c.upper(), c.lower(), deaccent(c), deaccent(c).upper(), deaccent(c).lower()):
            if cand and cand in vocab:
                out.append(vocab[cand])
                break
    return out


def words_from_emission(logp: np.ndarray, vocab: dict[str, int], words: list[str], seconds: float,
                        blank: int = 0, sep: bool = True) -> list[tuple[float, float] | None]:
    """El tiempo de cada palabra (None si no tiene letras que buscar o no se ha encontrado). Con `sep`, el separador de
    palabras del modelo («|») entre una y otra, como lo aprendió (los límites caen mejor)."""
    ids, owner = [], []
    bar = vocab.get("|") if sep else None
    for k, w in enumerate(words):
        c = char_ids(w, vocab)
        if c and ids and bar is not None:
            ids.append(bar)
            owner.append(-1)
        ids += c
        owner += [k] * len(c)
    out: list[tuple[float, float] | None] = [None] * len(words)
    if not ids:
        return out
    spans = ctc_align(logp, ids, blank)
    if spans is None:
        return out
    fd = seconds / max(1, logp.shape[0])
    for (a, b), k in zip(spans, owner):
        if k < 0:
            continue
        s, e = a * fd, (b + 1) * fd
        cur = out[k]
        out[k] = (s, e) if cur is None else (min(cur[0], s), max(cur[1], e))
    return out


def fill_times(times: list[tuple[float, float] | None], total: float) -> list[tuple[float, float]]:
    """Las palabras sin tiempo (un signo suelto, una que no se ha encontrado), en el hueco entre sus vecinas."""
    out: list[tuple[float, float]] = []
    n = len(times)
    for i, t in enumerate(times):
        if t is not None:
            s = max(t[0], out[-1][1] if out else 0.0)
            out.append((s, max(s, t[1])))
            continue
        prev = out[-1][1] if out else 0.0
        nxt = next((x[0] for x in times[i + 1:] if x is not None), total)
        run = next((j for j in range(i, n) if times[j] is not None), n) - i
        step = max(0.0, nxt - prev) / max(1, run)
        out.append((prev, prev + step))
    return [(round(a, 3), round(b, 3)) for a, b in out]


# ---------- modelos (en la GPU; nunca en el modo de prueba) ----------

def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _vram(name: str) -> None:
    """Pide sitio en la GPU (vram.ensure de núcleo: descarga los modelos de otras partes si hace falta)."""
    try:
        import vram  # type: ignore  (docker/gpu/vram.py, junto a server.py)
    except ImportError:
        try:
            from .. import vram  # type: ignore
        except Exception:  # noqa: BLE001
            return
    try:
        vram.ensure(name)
    except TypeError:
        try:
            vram.ensure(name, VRAM_GB.get(name, 10.0))
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        pass


def _kwargs_for(fn: Any, **kw: Any) -> dict[str, Any]:
    import inspect
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return kw
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return kw
    return {k: v for k, v in kw.items() if k in params}


def _chatterbox(pack: str = "") -> Any:
    key = pack or "mtl"
    if key in _M:
        return _M[key]
    _vram("tts")
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    if "mtl" not in _M:
        try:
            _M["mtl"] = ChatterboxMultilingualTTS.from_pretrained(device=_device(), t3_model="v3")
        except TypeError:   # (una versión del paquete sin V3: la que haya)
            NOTES.append("chatterbox sin t3_model: V2")
            _M["mtl"] = ChatterboxMultilingualTTS.from_pretrained(device=_device())
    if not pack:
        return _M["mtl"]
    try:   # el paquete de idioma: los pesos de su T3 sobre otro modelo V3 (mismo tokenizador y el mismo S3Gen V3)
        from huggingface_hub import hf_hub_download, list_repo_files
        from safetensors.torch import load_file
        repo = PACK_REPOS[pack]
        name = next(f for f in list_repo_files(repo) if f.startswith("t3_") and f.endswith(".safetensors"))
        m = ChatterboxMultilingualTTS.from_pretrained(device=_device(), t3_model="v3")
        sd = load_file(hf_hub_download(repo, name), device=str(next(m.t3.parameters()).device))
        m.t3.load_state_dict(sd, strict=True)
        _M[key] = m
    except Exception as e:  # noqa: BLE001  (sin el paquete, el modelo general de 23 idiomas)
        NOTES.append(f"paquete {pack}: {str(e)[:160]}")
        _M[key] = _M["mtl"]
    return _M[key]


def _asr() -> Any:
    if "asr" not in _M:
        _vram("tts")
        import torch
        from transformers import pipeline
        dev = _device()
        _M["asr"] = pipeline("automatic-speech-recognition", model=ASR_REPO, device=0 if dev == "cuda" else -1,
                             torch_dtype=torch.float16 if dev == "cuda" else torch.float32)
    return _M["asr"]


def _aligner(lang: str) -> tuple[Any, Any, dict[str, int], int]:
    key = f"align_{lang}"
    if key not in _M:
        _vram("tts")
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        repo = ALIGNERS.get(lang, ALIGNERS["en"])
        proc = Wav2Vec2Processor.from_pretrained(repo)
        model = Wav2Vec2ForCTC.from_pretrained(repo).to(_device()).eval()
        vocab = dict(proc.tokenizer.get_vocab())
        blank = int(proc.tokenizer.pad_token_id if proc.tokenizer.pad_token_id is not None else 0)
        _M[key] = (proc, model, vocab, blank)
    return _M[key]


def unload() -> None:
    """Suelta todos los modelos de esta parte (para que quepan los de otra etapa)."""
    _M.clear()
    try:
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def loaded() -> list[str]:
    return sorted(_M)


def transcribe(audios: list[np.ndarray], lang: str) -> list[str]:
    """Lo que dice cada audio (a 24 kHz), según Whisper."""
    if fake():
        raise RuntimeError("transcribe no se usa en el modo de prueba")
    asr = _asr()
    xs = [resample(a, SR, ASR_SR) for a in audios]
    gk = {"language": WHISPER_LANG.get(lang, "english"), "task": "transcribe"}
    with _ASR_LOCK:
        try:
            res = asr([{"raw": x, "sampling_rate": ASR_SR} for x in xs], batch_size=8, generate_kwargs=gk)
        except (TypeError, ValueError, KeyError):   # (otra versión de transformers: el audio tal cual, ya a 16 kHz)
            res = asr(xs, batch_size=8, generate_kwargs=gk)
    return [str((r or {}).get("text") or "") for r in (res if isinstance(res, list) else [res])]


def align_audio(audio: np.ndarray, sr: int, words: list[str], lang: str) -> list[tuple[float, float] | None]:
    """El tiempo de cada palabra en el audio (alineación forzada con wav2vec2)."""
    import torch
    proc, model, vocab, blank = _aligner(lang)
    x = resample(audio, sr, ASR_SR)
    with torch.inference_mode():
        iv = proc(x, sampling_rate=ASR_SR, return_tensors="pt").input_values.to(model.device)
        logp = torch.log_softmax(model(iv).logits[0].float(), dim=-1).cpu().numpy()
    return words_from_emission(logp, vocab, words, len(x) / ASR_SR, blank)


# ---------- las voces de serie (su referencia) ----------

def design_reference(spec: dict[str, Any]) -> tuple[np.ndarray, int]:
    """La voz de referencia de una voz de serie, diseñada desde su descripción (Qwen3-TTS-VoiceDesign). Se hace una vez:
    después se guarda (en el pod y, de vuelta, en Cinestoria). Si el paquete qwen-tts no está en este Python, con el de
    su entorno aparte (CINESTORIA_QWEN_PY: chatterbox y qwen-tts piden versiones distintas de transformers)."""
    if fake():
        return fake_reference(spec), SR
    _vram("diseno")
    out = cache_dir() / f"_diseno_{spec['id']}.wav"
    script = HERE / "disenar_voz.py"
    py = os.environ.get("CINESTORIA_QWEN_PY") or sys.executable
    args = [py, str(script), "--instruct", spec["design"], "--texto", spec["texto"], "--idioma", spec["qwen_lang"],
            "--semilla", str(int(spec.get("seed") or 0)), "--salida", str(out)]
    with _DESIGN_LOCK:
        r = subprocess.run(args, capture_output=True, text=True, timeout=900)
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"no se pudo diseñar la voz {spec['id']}: {(r.stderr or r.stdout)[-300:]}")
    x, sr = read_wav(out.read_bytes())
    out.unlink(missing_ok=True)
    return x, sr


def fake_reference(spec: dict[str, Any]) -> np.ndarray:
    seed = zlib.crc32(str(spec.get("id")).encode())
    rng = np.random.default_rng(seed)
    t = np.arange(int(8.0 * SR)) / SR
    f0 = 110 if spec.get("g") == "m" else 200
    x = 0.2 * np.sin(2 * np.pi * (f0 + 10 * rng.random()) * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    return x.astype(np.float32)


def reference(voice_id: str | None, ref_b64: str | None, design_seed: Any = None
              ) -> tuple[Path | None, str, bool, np.ndarray | None]:
    """La referencia de la voz: la que manda el cliente; si no, la guardada; si no, la diseña (una voz de serie). Devuelve
    (archivo, huella, nueva, audio): sin voz ni referencia, (None, "", False, None): la voz por defecto de Chatterbox.
    `design_seed`: vuelve a diseñar la voz de serie con esa semilla (python -m narra.voice voces-gpu --semilla)."""
    if ref_b64:
        x, sr = from_b64(ref_b64)
        x = resample(x, sr, SR)
        h = hashlib.sha1(wav_bytes(x)).hexdigest()[:16]
        path = cache_dir() / f"ref_{h}.wav"
        if not path.exists():
            path.write_bytes(wav_bytes(x))
        return path, h, False, x
    if not voice_id:
        return None, "", False, None
    spec = voice_spec(voice_id)
    if spec is None:
        raise BadRequest(f"voz desconocida: {voice_id}")
    path = cache_dir() / f"{voice_id}.wav"
    new = False
    redo = isinstance(design_seed, int) and not isinstance(design_seed, bool)
    if redo:
        spec = {**spec, "seed": int(design_seed)}
    if redo or not path.exists():
        with _REF_LOCK:
            if redo or not path.exists():
                x, sr = design_reference(spec)
                x = trim(resample(x, sr, SR), SR)[: int(12.0 * SR)]
                tmp = path.with_suffix(".part")
                tmp.write_bytes(wav_bytes(normalize(x, SR)))
                tmp.replace(path)
                new = True
    x, _ = read_wav(path.read_bytes())
    return path, hashlib.sha1(path.read_bytes()).hexdigest()[:16], new, x


# ---------- la síntesis de una frase ----------

def _seed(text: str, seed: Any, attempt: int) -> int:
    base = int(seed) if isinstance(seed, (int, float)) and not isinstance(seed, bool) else zlib.crc32(text.encode())
    return (base + 7919 * attempt) & 0x7FFFFFFF


def fake_unit(words: list[str], seed: int, drop: bool) -> tuple[np.ndarray, list[tuple[float, float]], str]:
    """Modo de prueba: cada palabra, un tono del largo de sus letras (con su tiempo exacto). `drop`: se come el final
    (lo que hace a veces la voz de verdad), para probar el control y los reintentos."""
    said = words[:max(1, int(len(words) * 0.6))] if drop and len(words) > 1 else list(words)
    rng = np.random.default_rng(seed)
    parts, times, t = [np.zeros(int(0.05 * SR), np.float32)], [], 0.05
    for k, w in enumerate(said):
        d = 0.08 + 0.055 * max(1, len(w))
        tt = np.arange(int(d * SR)) / SR
        f = 130 + 9 * (k % 11) + 20 * rng.random()
        env = np.minimum(1.0, np.minimum(tt, d - tt) / 0.012)
        parts.append((0.22 * np.sin(2 * np.pi * f * tt) * env).astype(np.float32))
        times.append((t, t + d))
        t += d
        g = 0.16 if re.search(r"[,;:]$", w) else 0.06
        parts.append(np.zeros(int(g * SR), np.float32))
        t += g
    parts.append(np.zeros(int(0.08 * SR), np.float32))
    return np.concatenate(parts), times, " ".join(said)


def real_unit(text: str, lang: str, pack: str, ref: Path | None, style: float, temperature: float,
              seed: int) -> np.ndarray:
    import torch
    m = _chatterbox(pack)
    ex = float(min(0.9, max(0.25, style)))
    kw = _kwargs_for(m.generate, language_id=lang, audio_prompt_path=str(ref) if ref else None, exaggeration=ex,
                     cfg_weight=0.5 if ex <= 0.55 else 0.35, temperature=float(min(1.0, max(0.4, temperature))))
    if kw.get("audio_prompt_path") is None:
        kw.pop("audio_prompt_path", None)
    with _LOCK, torch.inference_mode():
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        wav = m.generate(text, **kw)
    x = wav.squeeze().float().cpu().numpy() if hasattr(wav, "cpu") else np.asarray(wav, np.float32).squeeze()
    return resample(x, int(getattr(m, "sr", SR)), SR)


def check(raw: np.ndarray, ref_words: list[str], alt_words: list[str] | None, hyp: str, lang: str) -> dict[str, Any]:
    """El control de una frase (antes de igualar su nivel). ok, la proporción de palabras mal y por qué no vale."""
    why: list[str] = []
    x = np.asarray(raw, np.float32)
    if len(x) < int(0.1 * SR) or not np.isfinite(x).all():
        return {"ok": False, "wer": 1.0, "why": ["invalido"], "dur": 0.0}
    hw = norm_words(hyp, lang)
    rw = norm_words(" ".join(ref_words), lang)
    w_ = wer(rw, hw)
    if alt_words:
        w_ = min(w_, wer(norm_words(" ".join(alt_words), lang), hw))
    allowed = max(1.0 / max(1, len(rw)), WER_MAX) if len(rw) < 9 else WER_MAX
    if w_ > allowed + 1e-9:
        why.append("palabras")
    ends = {rw[-1]} if rw else set()
    if alt_words and norm_words(" ".join(alt_words), lang):
        ends.add(norm_words(" ".join(alt_words), lang)[-1])
    if len(rw) >= 3 and not ends & set(hw[-3:]):
        why.append("final")   # (se ha comido el final: lo que más le pasa a una voz así)
    a, b = voiced_span(x, SR)
    syl = syllables(rw)
    ratio = (b - a) / (syl / SYL_PER_S.get(lang, 5.0)) if syl >= 6 else 1.0
    if not DUR_RANGE[0] <= ratio <= DUR_RANGE[1]:
        why.append("duracion")
    if any(g > GAP_MAX for g in internal_gaps(x, SR)):
        why.append("silencio")
    if float(np.mean(np.abs(x) >= 0.995)) > CLIP_MAX:
        why.append("satura")
    f = frames_db(x, SR)
    top = float(f.max())
    v = f[f > top - 35]
    if top < QUIET_DB:
        why.append("callado")
    elif len(v) > 5 and float(np.max(v)) > float(np.percentile(v, 90)) + BURST_DB:
        why.append("ruido")
    return {"ok": not why, "wer": round(float(w_), 3), "why": why, "dur": round(float(ratio), 3)}


def _score(c: dict[str, Any]) -> float:
    return (0.0 if c["ok"] else 10.0) + c["wer"] + abs(math.log(max(0.05, c["dur"] or 1.0))) + 2 * len(c["why"])


def synth_sentence(toks: list[str], alt: str | None, lang: str, pack: str, ref: Path | None, style: float,
                   temperature: float, seed: Any, retries: int, do_check: bool, fake_fail: int = 0
                   ) -> tuple[np.ndarray, list[tuple[float, float]], dict[str, Any]]:
    """Una frase: su audio (al nivel de las demás), el tiempo de cada una de sus palabras (desde su principio) y su control.
    Si no pasa el control, otra vez con otra semilla (y algo más contenida); se queda la mejor."""
    spoken = [speakable(t, lang) for t in toks]
    owners = [k for k, (_, ws) in enumerate(spoken) for _ in ws]
    awords = [w for _, ws in spoken for w in ws]
    subs = sub_ranges([s for s, _ in spoken], 0, len(spoken))
    alt_words = alt.split() if alt and alt.split() != toks else None
    best = None
    tries = 1 + (max(0, int(retries)) if do_check else 0)
    for attempt in range(tries):
        sd = _seed(" ".join(toks), seed, attempt)
        st = style if attempt < 2 else 0.5 + (style - 0.5) * 0.5   # (el último intento, más contenido)
        temp = temperature if attempt == 0 else max(0.5, temperature - 0.1 * attempt)
        pieces, times, hyp_parts = [], [None] * len(awords), []
        t0 = 0.0
        for j, (a, b) in enumerate(subs):
            sub_words = [i for i, o in enumerate(owners) if a <= o < b]
            if fake():
                x, tw, hyp_j = fake_unit([awords[i] for i in sub_words], sd + j, attempt < fake_fail)
                for i, tt in zip(sub_words, tw):
                    times[i] = (t0 + tt[0], t0 + tt[1])
                hyp_parts.append(hyp_j)
            else:
                x = trim(real_unit(" ".join(s for s, _ in spoken[a:b]), lang, pack, ref, st, temp, sd + j), SR)
            pieces.append(x)
            t0 += len(x) / SR
            if j < len(subs) - 1:
                pieces.append(np.zeros(int(SUB_PAUSE * SR), np.float32))
                t0 += SUB_PAUSE
        raw = np.concatenate(pieces) if pieces else np.zeros(int(0.2 * SR), np.float32)
        if do_check:
            hyp = " ".join(hyp_parts) if fake() else transcribe([raw], lang)[0]
            c = check(raw, [w for w in awords], alt_words, hyp, lang)
        else:
            c = {"ok": True, "wer": 0.0, "why": [], "dur": 1.0}
        c["tries"] = attempt + 1
        cand = (raw, times, c)
        if best is None or _score(c) < _score(best[2]):
            best = cand
        if c["ok"]:
            break
    raw, times, c = best  # type: ignore[misc]
    c = {**c, "tries": tries if not c["ok"] else c["tries"]}
    if not fake():   # los tiempos de cada palabra, de la toma que se queda (alineación forzada)
        try:
            times = align_audio(raw, SR, awords, lang)
        except Exception as e:  # noqa: BLE001  (sin alineador: repartidas en la voz; el cliente lo ve en los tiempos)
            NOTES.append(f"alineación: {str(e)[:120]}")
            times = [None] * len(awords)
    audio = normalize(raw, SR)
    # el tiempo de cada palabra escrita: el de las palabras que se dicen por ella («1912» → «mil novecientos doce»)
    per_tok: list[tuple[float, float] | None] = [None] * len(toks)
    for i, k in enumerate(owners):
        t = times[i] if i < len(times) else None
        if t is not None:
            cur = per_tok[k]
            per_tok[k] = t if cur is None else (min(cur[0], t[0]), max(cur[1], t[1]))
    return audio, fill_times(per_tok, len(audio) / SR), c


# ---------- las rutas ----------

def _floats(v: Any, n: int, dflt: float) -> list[float]:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return [float(v)] * n
    if isinstance(v, list):
        out = []
        for i in range(n):
            try:
                out.append(float(v[i]))
            except (IndexError, TypeError, ValueError):
                out.append(dflt)
        return out
    return [dflt] * n


def tts(p: dict[str, Any]) -> dict[str, Any]:
    t_start = time.monotonic()
    if not isinstance(p, dict):
        raise BadRequest("se espera un objeto JSON")
    text = re.sub(r"\s+", " ", str(p.get("text") or "")).strip()
    if not text:
        raise BadRequest("falta el texto")
    lang = str(p.get("lang") or "es")[:2].lower()
    if lang not in LANGS:
        raise BadRequest(f"idioma no soportado: {lang}")
    voice_id = p.get("voice") or None
    spec = voice_spec(voice_id) if voice_id else None
    if voice_id and spec is None and not p.get("voice_ref"):
        raise BadRequest(f"voz desconocida: {voice_id}")
    if spec and spec.get("lang") != lang:
        raise BadRequest(f"la voz {voice_id} no es de {lang}")
    variant = str(p.get("variant") or (spec or {}).get("variant") or "")
    pack = (spec or {}).get("pack") or {"es-ES": "es-es", "es-419": "es-mx", "pt-BR": "pt-br",
                                        "pt-PT": "pt-pt"}.get(variant, "")
    ref, ref_h, ref_new, ref_x = reference(voice_id, p.get("voice_ref"), p.get("design_seed"))
    toks = text.split()
    ranges = chunk_ranges(toks, p.get("chunks"))
    n = len(ranges)
    styles = _floats(p.get("styles"), n, float(p.get("style") if isinstance(p.get("style"), (int, float)) else 0.5))
    pauses = _floats(p.get("pauses"), max(0, n - 1), PAUSE)
    alts = p.get("alts") if isinstance(p.get("alts"), list) else []
    retries = max(0, min(4, int(p.get("retries", 2) if isinstance(p.get("retries", 2), (int, float)) else 2)))
    do_check = bool(p.get("check", True))
    temperature = float(p.get("temperature") or 0.8)
    ff = p.get("fake_fail") if fake() and isinstance(p.get("fake_fail"), dict) else {}
    out = [np.zeros(int(LEAD * SR), np.float32)]
    t = LEAD
    words: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    for k, (a, b) in enumerate(ranges):
        alt = alts[k] if k < len(alts) and isinstance(alts[k], str) else None
        audio, times, c = synth_sentence(toks[a:b], alt, lang, pack, ref, styles[k], temperature, p.get("seed"),
                                         retries, do_check, int(ff.get(str(k), 0) or 0))
        for tok, (s, e) in zip(toks[a:b], times):
            words.append({"text": tok, "start": round(t + s, 3), "end": round(t + e, 3)})
        chunks.append({"text": " ".join(toks[a:b]), "start": round(t, 3), "end": round(t + len(audio) / SR, 3), **c})
        out.append(audio)
        t += len(audio) / SR
        if k < n - 1:
            gap = max(0.05, min(3.0, pauses[k]))
            out.append(np.zeros(int(gap * SR), np.float32))
            t += gap
    out.append(np.zeros(int(TAIL * SR), np.float32))
    audio = np.concatenate(out)
    resp: dict[str, Any] = {"audio": b64wav(audio), "sr": SR, "words": words, "duration": round(len(audio) / SR, 3),
                            "seconds": round(time.monotonic() - t_start, 3),
                            "model": f"{MODEL}/{pack or 'mtl'}" + ("/fake" if fake() else ""), "voice": voice_id,
                            "chunks": chunks}
    if ref_x is not None and not p.get("voice_ref"):
        resp["voice_ref"] = b64wav(ref_x)   # (para que Cinestoria la guarde: la misma voz en todos los pods)
        resp["voice_ref_new"] = ref_new
    return resp


def align(p: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(p, dict) or not p.get("audio"):
        raise BadRequest("falta el audio")
    text = re.sub(r"\s+", " ", str(p.get("text") or "")).strip()
    lang = str(p.get("lang") or "es")[:2].lower()
    if lang not in LANGS:
        raise BadRequest(f"idioma no soportado: {lang}")
    x, sr = from_b64(str(p["audio"]))
    toks = text.split()
    total = len(x) / sr
    spoken = [speakable(tk, lang) for tk in toks]
    owners = [k for k, (_, ws) in enumerate(spoken) for _ in ws]
    awords = [w for _, ws in spoken for w in ws]
    if fake():   # (las palabras repartidas en lo que suena, por sus letras)
        a, b = voiced_span(x, sr)
        lens = np.array([max(1, len(w)) for w in awords], float)
        edges = a + np.concatenate([[0], np.cumsum(lens)]) / max(1.0, lens.sum()) * max(0.0, b - a)
        times: list[tuple[float, float] | None] = [(float(edges[i]), float(edges[i + 1])) for i in range(len(awords))]
    else:
        times = align_audio(x, sr, awords, lang)
    per_tok: list[tuple[float, float] | None] = [None] * len(toks)
    for i, k in enumerate(owners):
        if times[i] is not None:
            cur = per_tok[k]
            per_tok[k] = times[i] if cur is None else (min(cur[0], times[i][0]), max(cur[1], times[i][1]))
    return {"words": [{"text": tk, "start": s, "end": e} for tk, (s, e) in zip(toks, fill_times(per_tok, total))]}


ROUTES = {"/tts": tts, "/align": align}


def _make_router() -> Any:
    try:
        from fastapi import APIRouter, Body, HTTPException
    except ImportError:
        return None
    r = APIRouter()

    def wrap(fn):
        def endpoint(payload: dict = Body(...)) -> dict:
            try:
                return fn(payload)
            except BadRequest as e:
                raise HTTPException(status_code=422, detail=str(e)) from e
        return endpoint

    for path, fn in ROUTES.items():
        r.add_api_route(path, wrap(fn), methods=["POST"])
    return r


router = _make_router()
