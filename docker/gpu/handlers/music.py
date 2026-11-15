"""1.30 (media): POST /music en la GPU del cliente («Tu GPU»).

Modelo: ACE-Step 1.5 XL-SFT (licencia MIT; https://huggingface.co/ACE-Step/acestep-v15-xl-sft) con su planificador
de 4B (acestep-5Hz-lm-4B, MIT). Es la variante de más calidad de ACE-Step 1.5 (la «XL» supera a Suno v5 en «Style
Align» en la tabla del propio equipo: 47,9 frente a 46,8) y se entrenó con música con licencia, de dominio público,
libre de derechos y sintética (MIDI), con uso comercial de lo que genera. Nunca MusicGen (pesos CC-BY-NC).

Contrato (contrato130.md):
  POST /music {"prompt": str, "seconds": float, "seed": int?, "bpm": int?, "instrumental": true}
       -> {"audio": b64wav, "sr": int, "seconds": float, "model": str}
  Además (campos de más, compatibles): la petición admite "keyscale" («D minor») y "timesignature" (3, 4…); la
  respuesta lleva "duration" (lo que dura el audio, s), "meta" (bpm, tonalidad y compás que usó el modelo) y "fake".
  "seconds" de la respuesta es el tiempo de GPU (como en /image).

- Importaciones pesadas (torch, acestep) solo dentro de las funciones: el servidor arranca sin ellas.
- Modo de prueba (CINESTORIA_FAKE=1): sin torch; una pista sintética (acordes con su caída final) de la duración
  pedida, estéreo a 48 kHz, la misma para la misma semilla.
- VRAM: antes de cargar, vram.ensure("musica") (el ayudante del servidor libera los modelos de otras etapas); unload()
  suelta el nuestro. Con 48 GB caben de sobra (XL en bf16 ~18,8 GB + el planificador 4B ~8 GB).
"""
from __future__ import annotations

import base64
import io
import math
import os
import tempfile
import threading
import time
from typing import Any

NAME = "musica"
SR = 48000
MAX_SECONDS = 300.0       # (Cinestoria nunca pide más de 300 s de una vez: sound.MUSIC_TRACK_MAX; el modelo llega a 600)
MIN_SECONDS = 10.0        # (ACE-Step compone de 10 a 600 s)
CAPTION_MAX = 512         # (ACE-Step: «caption», 512 caracteres como mucho)
DIT = os.environ.get("CINESTORIA_ACESTEP_DIT", "acestep-v15-xl-sft")
LM = os.environ.get("CINESTORIA_ACESTEP_LM", "acestep-5Hz-lm-4B")
STEPS = int(os.environ.get("CINESTORIA_ACESTEP_STEPS", "50"))      # (la SFT, con CFG: 50 pasos según su ficha)
GUIDANCE = float(os.environ.get("CINESTORIA_ACESTEP_CFG", "7.0"))
MODEL_ID = f"ace-step-1.5:{DIT}+{LM}"

# los pesos que necesita (los descarga y guarda models.py del servidor; repos públicos de HuggingFace)
MODELS: list[dict[str, Any]] = [
    {"name": "ace-step-1.5", "repo": "ACE-Step/Ace-Step1.5", "license": "MIT", "size_gb": 10.0,
     "why": "el paquete base de ACE-Step 1.5: VAE, codificador de texto y configuración"},
    {"name": DIT, "repo": f"ACE-Step/{DIT}", "license": "MIT", "size_gb": 18.8, "vram_gb": 20.0,
     "why": "el DiT XL-SFT (~4B): la variante de más calidad (50 pasos con CFG)"},
    {"name": LM, "repo": f"ACE-Step/{LM}", "license": "MIT", "size_gb": 8.0, "vram_gb": 9.0,
     "why": "el planificador (Qwen3 4B): estructura y códigos de 5 Hz; con 24 GB o más, el de más calidad"},
]
PIP = ["acestep @ git+https://github.com/ace-step/ACE-Step-1.5"]   # (fíjalo a un commit en requirements)
VRAM_GB = 30.0

_LOCK = threading.Lock()
_RUN = threading.Lock()   # (una generación a la vez en la GPU: las peticiones en paralelo esperan su turno)
_STATE: dict[str, Any] = {}


def fake() -> bool:
    return os.environ.get("CINESTORIA_FAKE", "") not in ("", "0", "false", "no")


def _vram():
    """El ayudante de VRAM del servidor (docker/gpu/vram.py), sea cual sea cómo se importe este módulo."""
    try:
        from .. import vram  # type: ignore[no-redef]
        return vram
    except Exception:  # noqa: BLE001
        pass
    try:
        import vram  # type: ignore[no-redef]
        return vram
    except Exception:  # noqa: BLE001
        return None


def _ensure_vram() -> None:
    v = _vram()
    if v is None:
        return
    if hasattr(v, "register"):
        try:
            v.register(NAME, unload)
        except Exception:  # noqa: BLE001
            pass
    if hasattr(v, "ensure"):
        v.ensure(NAME)


def clean_payload(p: dict[str, Any]) -> dict[str, Any]:
    """La petición, validada y con sus valores por defecto. ValueError si no vale (el servidor responde 422)."""
    if not isinstance(p, dict):
        raise ValueError("la petición debe ser un objeto JSON")
    prompt = str(p.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("falta prompt")
    try:
        seconds = float(p.get("seconds"))
    except (TypeError, ValueError):
        raise ValueError("seconds debe ser un número") from None
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds debe ser mayor que 0")
    seed = p.get("seed")
    seed = int(seed) % (2 ** 31) if isinstance(seed, (int, float)) and not isinstance(seed, bool) else None
    bpm = p.get("bpm")
    bpm = int(bpm) if isinstance(bpm, (int, float)) and not isinstance(bpm, bool) and 30 <= int(bpm) <= 300 else None
    ts = p.get("timesignature")
    ts = int(ts) if isinstance(ts, (int, float)) and int(ts) in (2, 3, 4, 6) else None
    key = str(p.get("keyscale") or "").strip()[:20]
    return {"prompt": prompt[:CAPTION_MAX], "seconds": min(MAX_SECONDS, max(MIN_SECONDS, seconds)), "seed": seed,
            "bpm": bpm, "timesignature": ts, "keyscale": key, "instrumental": p.get("instrumental", True) is not False}


def _wav_b64(x, sr: int) -> str:
    """float32 (n, 2) → WAV de 16 bits en base64 (sin soundfile: wave de la biblioteca estándar)."""
    import wave

    import numpy as np
    pcm = (np.clip(np.asarray(x, np.float32), -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(int(pcm.shape[1]) if pcm.ndim == 2 else 1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())
    return base64.b64encode(buf.getvalue()).decode()


FAKE_NOTES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
_FAKE_ALIAS = {"Db": 1, "D#": 3, "Gb": 6, "G#": 8, "A#": 10}


def fake_key(p: dict[str, Any]) -> tuple[int, bool, str]:
    """La tonalidad del modo de prueba: la pedida («D minor») o, sin ella, una según la semilla (como el planificador
    de ACE-Step, que la elige): (nota 0-11, menor, nombre)."""
    import re
    m = re.match(r"^\s*([A-G])([#b]?)\s*(major|minor|maj|min|m)?\s*$", p.get("keyscale") or "", re.I)
    if m:
        name = m.group(1).upper() + m.group(2)
        pc = FAKE_NOTES.index(name) if name in FAKE_NOTES else _FAKE_ALIAS.get(name, 0)
        minor = (m.group(3) or "major").lower() in ("minor", "min", "m")
    else:
        s = p["seed"] if p.get("seed") is not None else 7
        pc, minor = s % 12, (s // 12) % 2 == 0
    return pc, minor, f"{FAKE_NOTES[pc]} {'minor' if minor else 'major'}"


def _fake_audio(p: dict[str, Any]):
    """Una pista sintética del largo pedido en su tonalidad (i-VI-iv-V en menor, I-vi-IV-V en mayor), que acaba en el
    acorde de la tónica que se apaga, como una pieza de verdad. Sin torch."""
    import numpy as np
    pc, minor, _ = fake_key(p)
    n = int(round(p["seconds"] * SR))
    t = np.arange(n, dtype=np.float64) / SR
    tonic = 220.0 * 2 ** ((pc - 9) / 12)
    third = 3 if minor else 4
    prog = [(0, third), (8, 4), (5, 3), (7, 4)] if minor else [(0, 4), (9, 3), (5, 4), (7, 4)]
    bar = 60.0 / (p["bpm"] or 72) * (p["timesignature"] or 4)

    def chord(deg: int, th: int, tt):
        f0 = tonic * 2 ** (deg / 12)
        return sum(np.sin(2 * np.pi * f0 * 2 ** (s / 12) * tt) for s in (0, th, 7)) + 0.6 * np.sin(np.pi * f0 * tt)
    x = np.zeros(n)
    for k in range(int(p["seconds"] / (2 * bar)) + 1):
        a, b = int(k * 2 * bar * SR), min(n, int((k + 1) * 2 * bar * SR + SR))
        if a >= n:
            break
        tt = t[a:b] - t[a]
        env = np.minimum(1, tt / 0.8) * np.exp(-tt * 0.35)
        x[a:b] += 0.08 * env * chord(*prog[k % 4], tt)
    k = max(0, n - int(4.0 * SR))   # (el acorde final, 4 s antes del final, que se apaga)
    x[max(0, k - int(0.5 * SR)):k] *= 0.3
    tt = t[k:] - t[k]
    x[k:] = 0.16 * np.exp(-tt * 1.1) * chord(0, third, tt)
    st = np.stack([x, np.roll(x, int(0.011 * SR))], axis=1).astype(np.float32)
    return st, SR


def _load():
    """ACE-Step 1.5 (DiT XL-SFT y el planificador), una vez por pod."""
    with _LOCK:
        if _STATE.get("dit") is not None:
            return _STATE["dit"], _STATE["llm"]
        _ensure_vram()
        from acestep.handler import AceStepHandler  # type: ignore[import-not-found]
        from acestep.llm_inference import LLMHandler  # type: ignore[import-not-found]
        root = os.environ.get("CINESTORIA_ACESTEP_ROOT") or os.path.join(
            os.environ.get("CINESTORIA_MODELS", "/models"), "ace-step")
        ckpt = os.path.join(root, "checkpoints")
        dit = AceStepHandler()
        dit.initialize_service(project_root=root, config_path=DIT, device="cuda")
        llm = LLMHandler()
        # (el planificador con PyTorch, no vllm: vllm se reserva una parte fija de la VRAM y no la suelta para Wan)
        llm.initialize(checkpoint_dir=ckpt, lm_model_path=LM, backend=os.environ.get("CINESTORIA_ACESTEP_LM_BACKEND", "pt"),
                       device="cuda")
        _STATE.update(dit=dit, llm=llm)
        return dit, llm


def unload() -> None:
    """Suelta ACE-Step de la VRAM (lo llama vram.ensure cuando otra etapa necesita la memoria)."""
    with _LOCK:
        had = _STATE.pop("dit", None) is not None
        _STATE.pop("llm", None)
    if had and not fake():
        try:
            import gc

            import torch  # type: ignore[import-not-found]
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def _real_audio(p: dict[str, Any]):
    import numpy as np
    from acestep.inference import GenerationConfig, GenerationParams, generate_music  # type: ignore[import-not-found]
    dit, llm = _load()
    params = GenerationParams(
        task_type="text2music", caption=p["prompt"], lyrics="[Instrumental]", instrumental=bool(p["instrumental"]),
        bpm=p["bpm"], keyscale=p["keyscale"], timesignature=str(p["timesignature"] or ""), duration=float(p["seconds"]),
        inference_steps=STEPS, guidance_scale=GUIDANCE, seed=p["seed"] if p["seed"] is not None else -1,
        # el planificador piensa la estructura, pero no reescribe nuestra descripción ni pone letra
        thinking=True, use_cot_caption=False, use_cot_lyrics=False,
        use_cot_metas=not (p["bpm"] and p["keyscale"]))
    config = GenerationConfig(batch_size=1, use_random_seed=p["seed"] is None,
                              seeds=[p["seed"]] if p["seed"] is not None else None, audio_format="wav")
    with tempfile.TemporaryDirectory() as tmp:
        res = generate_music(dit, llm, params, config, save_dir=tmp)
    if not getattr(res, "success", False) or not getattr(res, "audios", None):
        raise RuntimeError(f"ACE-Step no compuso la pista: {getattr(res, 'error', '') or 'sin audio'}")
    a = res.audios[0]
    x = a["tensor"].float().cpu().numpy()   # [canales, muestras]
    sr = int(a.get("sample_rate") or SR)
    x = x.T if x.ndim == 2 else np.stack([x, x], axis=1)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    used = a.get("params") if isinstance(a.get("params"), dict) else {}
    meta = {k: used.get(k) or used.get("cot_" + k) or getattr(params, "cot_" + k, None)
            for k in ("bpm", "keyscale", "timesignature")}
    return x[:, :2].astype(np.float32), sr, meta


def music(payload: dict[str, Any]) -> dict[str, Any]:
    """POST /music."""
    p = clean_payload(payload)
    with _RUN:
        return _music(p)


def _music(p: dict[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    if fake():
        x, sr = _fake_audio(p)
        meta: dict[str, Any] = {"bpm": p["bpm"] or 72, "keyscale": fake_key(p)[2], "timesignature": p["timesignature"] or 4}
        model = "fake:" + MODEL_ID
    else:
        x, sr, meta = _real_audio(p)
        model = MODEL_ID
        meta = {"bpm": p["bpm"] or meta.get("bpm"), "keyscale": p["keyscale"] or meta.get("keyscale"),
                "timesignature": p["timesignature"] or meta.get("timesignature")}
    return {"audio": _wav_b64(x, sr), "sr": int(sr), "seconds": round(time.perf_counter() - t0, 3), "model": model,
            "duration": round(len(x) / sr, 3), "meta": meta, "fake": fake()}


ROUTES = {"/music": music}   # (para un servidor que monta funciones)
try:   # (y un router de FastAPI; la autenticación la pone el servidor en todas las rutas)
    from fastapi import APIRouter, HTTPException

    router = APIRouter()

    @router.post("/music")
    def _music_route(payload: dict) -> dict:   # noqa: D401
        try:
            return music(payload)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
except Exception:  # noqa: BLE001  (sin FastAPI: solo ROUTES)
    router = None
