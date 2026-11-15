"""1.30 (media): POST /clip en la GPU del cliente («Tu GPU»): imagen a vídeo.

Modelo: Wan 2.2 I2V-A14B (Apache 2.0; https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers; «We claim no rights
over your generated contents»), el mejor imagen a vídeo abierto con licencia comercial de verdad en noviembre de 2026
(ver NOTAS.md: HunyuanVideo 1.5 excluye la UE, LTX-2.x y MiniMax H3 llevan licencias con topes o territorios). Con la
destilación Wan2.2-Lightning de LightX2V (Apache 2.0): 4 pasos sin CFG en vez de 40 con CFG (unas 20 veces menos
cálculo; según su ficha, la misma calidad que el modelo base en la mayoría de los casos). "steps" ≥ 8 en la petición
usa el modelo base, sin la destilación (la referencia de calidad, unas 20 veces más lento).

Cadena de cada clip (el contrato pide el tamaño y los fotogramas del vídeo final):
  1. Wan 2.2 a 720p (1280×720 o 720×1280, el área de su entrenamiento) a 16 fps: hasta 81 fotogramas (5 s, lo que
     entrenó); un plano más largo lo cubre el montaje (render.clip_stretch, como mucho un 30 % más lento).
  2. Ampliación ×2 con Real-ESRGAN x2plus (BSD-3-Clause), mezclada con la ampliación Lanczos (que no quede ese
     enfoque de IA), y recorte al tamaño pedido (1920×1080 o 1080×1920).
  3. Interpolación a los fps pedidos (30) con RIFE 4.25 (MIT); sin RIFE, minterpolate de ffmpeg (compensación de
     movimiento). Nunca fotogramas repetidos: un clip de 16 fps duplicado a 30 se nota a tirones.
  4. H.264 de alta calidad (CRF 14), sin audio.

Contrato (contrato130.md):
  POST /clip {"image": b64png, "prompt": str, "seconds": float, "width": int, "height": int, "fps": int?, "seed": int?}
       -> {"video": b64mp4, "seconds": float, "model": str}
  Además (campos de más, compatibles): "negative" y "steps" en la petición; en la respuesta, "duration" (lo que dura el
  vídeo), "frames", "fps", "width", "height", "upscale", "interp" y "fake". "seconds" de la respuesta es el tiempo de
  GPU (como en /image).

VRAM: los dos expertos de Wan 2.2 A14B en bf16 son 2 × 28,6 GB y el codificador UMT5-XXL unos 11 GB: con 80 GB o más
(A100 80 GB, H100, RTX PRO 6000 de 96 GB) va en bf16 con descarga de los modelos que no se usan a la memoria del pod;
con menos (A40, L40S, A6000 de 48 GB), los expertos se guardan en fp8 (se calcula en bf16) y se descargan igual: algo
más lento y un poco menos fino. Modo de prueba (CINESTORIA_FAKE=1): sin torch, un vídeo de verdad (la imagen con un
acercamiento lento) del tamaño, la duración y los fps pedidos.
"""
from __future__ import annotations

import base64
import io
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

NAME = "clips"
WAN_REPO = os.environ.get("CINESTORIA_WAN_REPO", "Wan-AI/Wan2.2-I2V-A14B-Diffusers")
LIGHTNING_REPO = os.environ.get("CINESTORIA_WAN_LORA_REPO", "lightx2v/Wan2.2-Lightning")
LIGHTNING_DIR = "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1"
LORA_HIGH = os.environ.get("CINESTORIA_WAN_LORA_HIGH", f"{LIGHTNING_DIR}/high_noise_model.safetensors")
LORA_LOW = os.environ.get("CINESTORIA_WAN_LORA_LOW", f"{LIGHTNING_DIR}/low_noise_model.safetensors")
ESRGAN_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
RIFE_REPO = "daydreamlive/RIFE"   # (espejo de los pesos de Practical-RIFE 4.25, MIT)
WAN_FPS = 16
MAX_FRAMES = 81                   # (5 s a 16 fps: lo que entrenó Wan 2.2)
AREA_720 = 1280 * 720
FAST_STEPS = 4                    # (Lightning: 4 pasos, sin CFG)
BASE_STEPS, BASE_CFG = 40, 3.5    # (el modelo base: 40 pasos con CFG 3,5, el ejemplo de su ficha)
ESRGAN_MIX = float(os.environ.get("CINESTORIA_ESRGAN_MIX", "0.6"))   # (0 = solo Lanczos, 1 = solo Real-ESRGAN)
VRAM_GB_TOP = 80.0                # bf16 sin perder nada; con menos, fp8
VRAM_GB = 44.0                    # (lo mínimo, en fp8 con descarga)
CRF = 14
NEG_DEFAULT = ("blurry, low quality, distorted, morphing, warping, flicker, jitter, text, watermark, subtitles, "
               "static frame, fast motion, time-lapse, extra fingers, deformed hands, deformed face")
MODEL_TOP = "wan2.2-i2v-a14b+lightx2v-4step"
MODEL_BASE = "wan2.2-i2v-a14b"

MODELS: list[dict[str, Any]] = [
    {"name": "wan2.2-i2v-a14b", "repo": WAN_REPO, "license": "Apache-2.0", "size_gb": 70.0, "vram_gb": VRAM_GB_TOP,
     "why": "imagen a vídeo (dos expertos MoE de 14B, UMT5-XXL y VAE de Wan 2.2), formato Diffusers"},
    {"name": "wan2.2-lightning-i2v", "repo": LIGHTNING_REPO, "allow": [f"{LIGHTNING_DIR}/*"], "license": "Apache-2.0",
     "size_gb": 1.2, "why": "la destilación a 4 pasos (LoRA de rango 64 para cada experto)"},
    {"name": "realesrgan-x2plus", "url": ESRGAN_URL, "license": "BSD-3-Clause", "size_gb": 0.07,
     "why": "la ampliación ×2 de 720p al tamaño del vídeo"},
    {"name": "rife-4.25", "repo": RIFE_REPO, "license": "MIT", "size_gb": 0.03, "optional": True,
     "code": "https://github.com/hzwer/Practical-RIFE (MIT; carpeta en CINESTORIA_RIFE_DIR)",
     "why": "de 16 fps a los del vídeo sin fotogramas repetidos (sin él, minterpolate de ffmpeg)"},
]
PIP = ["diffusers>=0.35", "transformers", "accelerate", "peft", "ftfy", "spandrel"]

_LOCK = threading.Lock()
_RUN = threading.Lock()   # (una generación a la vez en la GPU: las peticiones en paralelo esperan su turno)
_STATE: dict[str, Any] = {}


def fake() -> bool:
    return os.environ.get("CINESTORIA_FAKE", "") not in ("", "0", "false", "no")


def _vram():
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


def _even(v: Any, lo: int = 64, hi: int = 4096) -> int:
    n = int(v)
    if not lo <= n <= hi:
        raise ValueError(f"tamaño fuera de rango: {n}")
    return n - n % 2


def clean_payload(p: dict[str, Any]) -> dict[str, Any]:
    """La petición, validada. ValueError si no vale (el servidor responde 422)."""
    if not isinstance(p, dict):
        raise ValueError("la petición debe ser un objeto JSON")
    raw = p.get("image")
    if not isinstance(raw, str) or len(raw) < 32:
        raise ValueError("falta image (PNG en base64)")
    try:
        img = base64.b64decode(raw.split(",", 1)[-1], validate=False)
    except Exception:  # noqa: BLE001
        raise ValueError("image no es base64") from None
    try:
        seconds = float(p.get("seconds"))
    except (TypeError, ValueError):
        raise ValueError("seconds debe ser un número") from None
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds debe ser mayor que 0")
    try:
        w, h = _even(p.get("width")), _even(p.get("height"))
    except (TypeError, ValueError) as e:
        raise ValueError(f"width/height: {e}") from None
    fps = p.get("fps")
    fps = int(fps) if isinstance(fps, (int, float)) and not isinstance(fps, bool) and 8 <= int(fps) <= 60 else 30
    seed = p.get("seed")
    seed = int(seed) % (2 ** 31) if isinstance(seed, (int, float)) and not isinstance(seed, bool) else 0
    steps = p.get("steps")
    steps = int(steps) if isinstance(steps, (int, float)) and not isinstance(steps, bool) and 1 <= int(steps) <= 80 \
        else FAST_STEPS
    return {"image": img, "prompt": str(p.get("prompt") or "").strip()[:2000], "seconds": min(10.0, max(0.5, seconds)),
            "width": w, "height": h, "fps": fps, "seed": seed, "steps": steps,
            "negative": str(p.get("negative") or "").strip()[:800]}


def wan_frames(seconds: float) -> int:
    """Los fotogramas que se piden a Wan (4k+1, a 16 fps) para `seconds` s: los que cubren el plano, hasta 81."""
    need = int(math.ceil(seconds * WAN_FPS)) + 1
    k = max(1, int(math.ceil((need - 1) / 4)))
    return min(MAX_FRAMES, 4 * k + 1)


def wan_size(w: int, h: int, mod: int = 16) -> tuple[int, int]:
    """El tamaño de generación: el área de 720p con la forma del vídeo, múltiplo de `mod`."""
    ar = h / w
    gh = int(round(math.sqrt(AREA_720 * ar))) // mod * mod
    gw = int(round(math.sqrt(AREA_720 / ar))) // mod * mod
    return gw, gh


def out_frames(n_src: int, fps: int) -> int:
    """Los fotogramas del vídeo final: lo que duran los de Wan, a los fps pedidos."""
    return max(1, int(round((n_src - 1) / WAN_FPS * fps)) + 1)


def _encode(frames, fps: int, path: str) -> None:
    """Fotogramas RGB (PIL o uint8 H×W×3) → H.264 por ffmpeg (tubería)."""
    import numpy as np
    first = np.asarray(frames[0])
    h, w = first.shape[:2]
    ff = shutil.which("ffmpeg") or "ffmpeg"
    p = subprocess.Popen([ff, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
                          "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", str(CRF), "-pix_fmt",
                          "yuv420p", "-movflags", "+faststart", path], stdin=subprocess.PIPE)
    try:
        for f in frames:
            p.stdin.write(np.ascontiguousarray(np.asarray(f, np.uint8)).tobytes())
    finally:
        p.stdin.close()
        if p.wait() != 0:
            raise RuntimeError("ffmpeg no pudo escribir el clip")


def _cover(img, w: int, h: int):
    """La imagen llenando w×h (escala y recorte centrado), como hace el montaje."""
    from PIL import Image
    sw, sh = img.size
    s = max(w / sw, h / sh)
    nw, nh = max(w, int(round(sw * s))), max(h, int(round(sh * s)))
    img = img.resize((nw, nh), Image.LANCZOS)
    x, y = (nw - w) // 2, (nh - h) // 2
    return img.crop((x, y, x + w, y + h))


def _fake_frames(p: dict[str, Any]) -> tuple[list, int]:
    """El modo de prueba: la imagen con un acercamiento lento (un 6 %) y una leve deriva, a los fps pedidos."""
    from PIL import Image
    img = Image.open(io.BytesIO(p["image"])).convert("RGB")
    w, h = p["width"], p["height"]
    base = _cover(img, int(w * 1.1) + 2, int(h * 1.1) + 2)
    n_src = wan_frames(p["seconds"])
    n = out_frames(n_src, p["fps"])
    frames = []
    for i in range(n):
        u = i / max(1, n - 1)
        z = 1.0 + 0.06 * u
        cw, ch = base.width / z, base.height / z
        x0 = (base.width - cw) / 2 + (base.width - cw) * 0.15 * (u - 0.5)
        y0 = (base.height - ch) / 2
        frames.append(base.resize((w, h), Image.BILINEAR, box=(x0, y0, x0 + cw, y0 + ch)))
    return frames, n_src


# ---------- el camino de verdad (GPU) ----------

def _pipe():
    """Wan 2.2 I2V-A14B (Diffusers) con la destilación de 4 pasos cargada en los dos expertos, una vez por pod."""
    with _LOCK:
        if _STATE.get("pipe") is not None:
            return _STATE["pipe"]
        _ensure_vram()
        import torch  # type: ignore[import-not-found]
        from diffusers import WanImageToVideoPipeline  # type: ignore[import-not-found]
        vram = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        pipe = WanImageToVideoPipeline.from_pretrained(WAN_REPO, torch_dtype=torch.bfloat16)
        try:
            pipe.load_lora_weights(LIGHTNING_REPO, weight_name=LORA_HIGH, adapter_name="lightning")
            pipe.load_lora_weights(LIGHTNING_REPO, weight_name=LORA_LOW, adapter_name="lightning_2",
                                   load_into_transformer_2=True)
            _STATE["lora"] = True
        except Exception as e:  # noqa: BLE001  (sin la destilación: solo el modelo base, si se pide con steps ≥ 8)
            _STATE["lora"] = False
            _STATE["lora_error"] = str(e)[:200]
        if vram < VRAM_GB_TOP - 4:   # (48 GB: los expertos guardados en fp8, el cálculo en bf16)
            for tr in (pipe.transformer, getattr(pipe, "transformer_2", None)):
                if tr is not None:
                    tr.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
            _STATE["precision"] = "fp8"
        else:
            _STATE["precision"] = "bf16"
        if vram < 90:   # (en la GPU solo el modelo que trabaja; el resto, en la memoria del pod)
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        _STATE["pipe"] = pipe
        return pipe


def _esrgan():
    if "esrgan" in _STATE:
        return _STATE["esrgan"]
    try:
        import torch  # type: ignore[import-not-found]
        from spandrel import ModelLoader  # type: ignore[import-not-found]
        path = os.path.join(os.environ.get("CINESTORIA_MODELS", "/models"), "realesrgan-x2plus", "RealESRGAN_x2plus.pth")
        m = ModelLoader().load_from_file(path).model.eval().half().to("cuda")
        _STATE["esrgan"] = (m, torch)
    except Exception:  # noqa: BLE001  (sin el modelo, Lanczos)
        _STATE["esrgan"] = None
    return _STATE["esrgan"]


def _rife():
    if "rife" in _STATE:
        return _STATE["rife"]
    try:
        d = os.environ.get("CINESTORIA_RIFE_DIR", "/opt/Practical-RIFE")
        if d not in sys.path:
            sys.path.insert(0, d)
        from train_log.RIFE_HDv3 import Model  # type: ignore[import-not-found]
        m = Model()
        m.load_model(os.path.join(d, "train_log"), -1)
        m.eval()
        m.device()
        _STATE["rife"] = m
    except Exception:  # noqa: BLE001  (sin RIFE, minterpolate)
        _STATE["rife"] = None
    return _STATE["rife"]


def _upscale(frames, w: int, h: int) -> tuple[list, str]:
    """De 720p al tamaño del vídeo: Real-ESRGAN ×2 mezclado con Lanczos (ESRGAN_MIX) y recorte; sin el modelo, Lanczos."""
    import numpy as np
    from PIL import Image
    m = _esrgan()
    out = []
    for f in frames:
        img = f if isinstance(f, Image.Image) else Image.fromarray(np.asarray(f, np.uint8))
        lz = _cover(img, w, h)
        if m is None or ESRGAN_MIX <= 0:
            out.append(lz)
            continue
        net, torch = m
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(img, np.float32) / 255.0).permute(2, 0, 1)[None].half().to("cuda")
            y = net(x).clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy()
        up = _cover(Image.fromarray((y * 255.0 + 0.5).astype(np.uint8)), w, h)
        out.append(Image.blend(lz, up, ESRGAN_MIX))
    return out, ("realesrgan-x2+lanczos" if m is not None and ESRGAN_MIX > 0 else "lanczos")


def _interpolate(frames, fps: int, tmpdir: str) -> tuple[list | None, str, str | None]:
    """De 16 fps a `fps`: RIFE con instantes intermedios exactos; sin RIFE, minterpolate de ffmpeg (devuelve la ruta
    del vídeo ya escrito)."""
    import numpy as np
    n_out = out_frames(len(frames), fps)
    m = _rife()
    if m is not None:
        import torch  # type: ignore[import-not-found]
        import torch.nn.functional as F  # type: ignore[import-not-found]
        arr = [np.asarray(f, np.uint8) for f in frames]
        h, w = arr[0].shape[:2]
        ph, pw = ((h - 1) // 128 + 1) * 128, ((w - 1) // 128 + 1) * 128

        def ten(a):
            t = torch.from_numpy(a).permute(2, 0, 1)[None].float().cuda() / 255.0
            return F.pad(t, (0, pw - w, 0, ph - h))
        out = []
        cache: dict[int, Any] = {}
        for j in range(n_out):
            pos = j * WAN_FPS / fps
            i0 = min(len(arr) - 1, int(math.floor(pos)))
            t = pos - i0
            if t < 1e-3 or i0 >= len(arr) - 1:
                out.append(arr[i0])
                continue
            for k in (i0, i0 + 1):
                if k not in cache:
                    cache = {q: v for q, v in cache.items() if q >= i0}
                    cache[k] = ten(arr[k])
            with torch.no_grad():
                mid = m.inference(cache[i0], cache[i0 + 1], timestep=float(t), scale=1.0)
            out.append((mid[0].clamp(0, 1) * 255.0 + 0.5).byte().permute(1, 2, 0).cpu().numpy()[:h, :w])
        return out, "rife-4.25", None
    src = os.path.join(tmpdir, "src16.mp4")
    dst = os.path.join(tmpdir, "interp.mp4")
    _encode(frames, WAN_FPS, src)
    ff = shutil.which("ffmpeg") or "ffmpeg"
    r = subprocess.run([ff, "-v", "error", "-y", "-i", src, "-vf",
                        f"minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1", "-an", "-c:v",
                        "libx264", "-preset", "medium", "-crf", str(CRF), "-pix_fmt", "yuv420p", "-movflags",
                        "+faststart", dst], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg no pudo interpolar el clip")
    return None, "minterpolate", dst


def _real_frames(p: dict[str, Any]) -> tuple[list, int, str]:
    import torch  # type: ignore[import-not-found]
    from PIL import Image
    pipe = _pipe()
    img = Image.open(io.BytesIO(p["image"])).convert("RGB")
    mod = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    gw, gh = wan_size(p["width"], p["height"], mod)
    img = _cover(img, gw, gh)
    n = wan_frames(p["seconds"])
    if p["steps"] < 8 and not _STATE.get("lora"):   # (sin la destilación, cada clip costaría 20 veces más: se dice ya y
        raise RuntimeError("no se pudo cargar la destilación de 4 pasos de Wan 2.2: "   # el vídeo sigue con su respaldo)
                           + str(_STATE.get("lora_error") or "")[:160])
    fast = p["steps"] < 8
    if _STATE.get("lora"):
        if fast:
            pipe.enable_lora()
            pipe.set_adapters(["lightning", "lightning_2"], adapter_weights=[1.0, 1.0])
        else:
            pipe.disable_lora()
    gen = torch.Generator(device="cuda").manual_seed(p["seed"])
    out = pipe(image=img, prompt=p["prompt"], negative_prompt=p["negative"] or NEG_DEFAULT, height=gh, width=gw,
               num_frames=n, guidance_scale=1.0 if fast else BASE_CFG, guidance_scale_2=1.0 if fast else BASE_CFG,
               num_inference_steps=p["steps"] if fast else max(p["steps"], BASE_STEPS), generator=gen,
               output_type="pil").frames[0]
    model = (MODEL_TOP if fast else MODEL_BASE) + f" ({_STATE.get('precision', 'bf16')})"
    return list(out), n, model


def unload() -> None:
    """Suelta Wan (y los ayudantes) de la VRAM: lo llama vram.ensure cuando otra etapa necesita la memoria."""
    with _LOCK:
        had = any(_STATE.pop(k, None) is not None for k in ("pipe", "esrgan", "rife"))
        _STATE.pop("lora", None)
    if had and not fake():
        try:
            import gc

            import torch  # type: ignore[import-not-found]
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def clip(payload: dict[str, Any]) -> dict[str, Any]:
    """POST /clip."""
    p = clean_payload(payload)
    with _RUN:
        return _clip(p)


def _clip(p: dict[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "clip.mp4")
        if fake():
            frames, n_src = _fake_frames(p)
            _encode(frames, p["fps"], path)
            model, up, interp = "fake:" + MODEL_TOP, "lanczos", "fake"
        else:
            src, n_src, model = _real_frames(p)
            big, up = _upscale(src, p["width"], p["height"])
            frames, interp, done = _interpolate(big, p["fps"], tmp)
            if done:
                shutil.move(done, path)
            else:
                _encode(frames, p["fps"], path)
        data = open(path, "rb").read()
    n = out_frames(n_src, p["fps"])
    return {"video": base64.b64encode(data).decode(), "seconds": round(time.perf_counter() - t0, 3), "model": model,
            "duration": round(n / p["fps"], 3), "frames": n, "fps": p["fps"], "width": p["width"],
            "height": p["height"], "upscale": up, "interp": interp, "fake": fake()}


ROUTES = {"/clip": clip}
try:
    from fastapi import APIRouter, HTTPException

    router = APIRouter()

    @router.post("/clip")
    def _clip_route(payload: dict) -> dict:   # noqa: D401
        try:
            return clip(payload)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
except Exception:  # noqa: BLE001
    router = None
