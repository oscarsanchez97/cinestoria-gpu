"""1.30 (imagen): las imágenes de Cinestoria en la GPU del cliente (RunPod), con el mejor modelo abierto que permite el uso
comercial. POST /image según el contrato de «Tu GPU» (contrato130):

    {"prompt": str, "negative": str?, "width": int, "height": int, "seed": int?, "steps": int?,
     "refs": [b64png]?, "ref_strength": float?, "n": int = 1}
    -> {"images": [b64png], "seconds": float, "model": str}

Lo que se usa (ver NOTAS.md de la 1.30, con las licencias citadas de cada ficha de modelo):
- Texto a imagen: Qwen-Image-2512 (Qwen, 20B, Apache 2.0), en bf16, a su tamaño nativo (16:9 = 1664 × 928), 40 pasos
  con CFG real 4,0 y su prompt negativo. Con `steps` ≤ 8, el LoRA Lightning de 8 pasos de lightx2v (Apache 2.0), CFG 1.
- Con `refs` (las fichas de personajes y sitios de Cinestoria, o la imagen que se edita): Qwen-Image-Edit-2511 (Qwen,
  20B, Apache 2.0), que comparte el codificador de texto y el VAE con el anterior (solo se carga su transformer).
- Ampliación: Real-ESRGAN x2plus (BSD-3-Clause) cuando lo pedido es mayor que el tamaño nativo: ×2 y después Lanczos al
  tamaño exacto (1920 × 1088 para 16:9, 1088 × 1920 para Shorts). Con `hd`, además la versión ×2 (para los detalles).

Extensiones al contrato (opcionales, sin ellas todo es como dice el contrato): "format": "png" | "jpeg" (las imágenes
de vuelta; jpeg ocupa una sexta parte por el proxy), "quality" (jpeg), "hd": bool (devuelve también "images_hd", la
ampliación ×2 a 3072 px de lado como mucho), "edit": bool (la única referencia es la imagen que se edita: se lee
entera; sin él, las referencias son fichas y se leen a ~0,6 MP). "ref_strength" se acepta y no se usa (Qwen-Image-Edit
no tiene ese ajuste: lo que se toma de cada ficha lo dice el prompt). La respuesta añade "native": [w, h], "upscaled": bool, "steps": int y, en
modo de prueba, "fake": true.

Los imports pesados (torch, diffusers, spandrel) van dentro de las funciones. Con CINESTORIA_FAKE=1 no se carga ningún
modelo: salen imágenes sintéticas del tamaño pedido (para probar el contrato en CPU).
"""
from __future__ import annotations

import base64
import hashlib
import importlib
import io
import math
import os
import threading
import time
from typing import Any

from PIL import Image, ImageDraw

# ---------- modelos ----------

T2I_REPO = "Qwen/Qwen-Image-2512"
EDIT_REPO = "Qwen/Qwen-Image-Edit-2511"
T2I_LORA = ("lightx2v/Qwen-Image-2512-Lightning", "Qwen-Image-2512-Lightning-8steps-V1.0-bf16.safetensors")
EDIT_LORA = ("lightx2v/Qwen-Image-Edit-2511-Lightning", "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors")
UPSCALER_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
UPSCALER_FILE = "RealESRGAN_x2plus.pth"

T2I_NAME, EDIT_NAME, UPSCALER_NAME = "qwen-image-2512", "qwen-image-edit-2511", "realesrgan-x2plus"

# Lo que models.py (nucleo) baja y guarda en la caché de pesos. "lazy": solo cuando se necesita por primera vez (el
# modelo de edición solo hace falta en los vídeos con fichas). Tamaños medidos en las páginas de Hugging Face.
MODELS: list[dict[str, Any]] = [
    {"name": T2I_NAME, "repo": T2I_REPO, "licence": "Apache-2.0", "gb": 57.7,
     "allow_patterns": ["model_index.json", "scheduler/*", "text_encoder/*", "tokenizer/*", "transformer/*", "vae/*"]},
    {"name": EDIT_NAME, "repo": EDIT_REPO, "licence": "Apache-2.0", "gb": 41.0, "lazy": True,
     "allow_patterns": ["model_index.json", "scheduler/*", "processor/*", "tokenizer/*", "transformer/*"]},
    {"name": "qwen-image-2512-lightning-8", "repo": T2I_LORA[0], "licence": "Apache-2.0", "gb": 0.85, "lazy": True,
     "allow_patterns": [T2I_LORA[1]]},
    {"name": "qwen-image-edit-2511-lightning-8", "repo": EDIT_LORA[0], "licence": "Apache-2.0", "gb": 0.85,
     "lazy": True, "allow_patterns": [EDIT_LORA[1]]},
    {"name": UPSCALER_NAME, "url": UPSCALER_URL, "file": UPSCALER_FILE, "licence": "BSD-3-Clause", "gb": 0.07},
]
VRAM_GB = {T2I_NAME: 58.0, EDIT_NAME: 41.0, UPSCALER_NAME: 0.2}   # residentes en bf16 (el de edición, solo su transformer)
MIN_VRAM_GB = 75.0   # una GPU de 80 GB (A100 80GB, H100, H100 NVL, RTX PRO 6000, H200): Qwen-Image en bf16 no cabe en 48

# ---------- tamaños ----------

# Los tamaños nativos de Qwen-Image-2512 (su ficha): a ellos se genera; lo demás, ampliando o reduciendo después
NATIVE = [(1664, 928), (928, 1664), (1328, 1328), (1472, 1104), (1104, 1472), (1584, 1056), (1056, 1584)]
UPSCALE_OVER = 1.04   # si lo pedido supera lo nativo en más de un 4 % por algún lado, se amplía con Real-ESRGAN ×2
HD_SIDE = 3072        # la versión ×2 que se devuelve con "hd" (= nitidez.HD_SIDE)
MAX_SIDE = 4096
MAX_N = 4
MAX_REFS = 4
MAX_PROMPT_CHARS = 6000
MAX_SEQ = 1024        # tokens del prompt que lee el codificador (diffusers corta en 512 si no se le dice)
EDIT_REF_AREA = 768 * 768   # las fichas entran al modelo de edición a ~0,6 MP (diffusers: 1 MP): la cara y la ropa se
                            # siguen viendo y cada imagen con fichas tarda ~2× una sin ellas (con 1 MP, ~2,5×)

DEFAULT_STEPS = 40    # (la ficha usa 50; 40 es lo habitual con la 2512 y cuesta un 20 % menos)
FAST_STEPS = 8        # con el LoRA Lightning
TRUE_CFG = 4.0
# El negativo que recomienda la ficha de Qwen-Image-2512 (en chino: baja resolución y calidad, miembros y dedos
# deformes, saturación excesiva, aspecto de cera, caras sin detalle, demasiado liso, aspecto de IA, composición caótica,
# texto borroso o deformado), con su traducción
NEGATIVE_BASE = ("低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。 "
                 "low resolution, low quality, deformed limbs, deformed fingers, oversaturated, waxy skin, faces without "
                 "detail, over-smoothed, AI-generated look, chaotic composition, blurry or distorted text")
LIGHTNING_SCHEDULER = {   # (el de generate_with_diffusers.py de Qwen-Image-Lightning: shift 3 en la destilación)
    "base_image_seq_len": 256, "base_shift": math.log(3), "invert_sigmas": False, "max_image_seq_len": 8192,
    "max_shift": math.log(3), "num_train_timesteps": 1000, "shift": 1.0, "shift_terminal": None,
    "stochastic_sampling": False, "time_shift_type": "exponential", "use_beta_sigmas": False,
    "use_dynamic_shifting": True, "use_exponential_sigmas": False, "use_karras_sigmas": False}

_LOCK = threading.Lock()   # una imagen a la vez en la GPU (las peticiones que llegan juntas esperan su turno)
_S: dict[str, Any] = {"t2i": None, "t2i_tr": None, "edit": None, "edit_tr": None, "up": None, "lora": {}}


def fake() -> bool:
    return os.environ.get("CINESTORIA_FAKE", "") not in ("", "0", "false", "no")


class BadRequest(ValueError):
    """Una petición que no cumple el contrato (422): no se reintenta."""


# ---------- la petición ----------

def _int(v: Any, name: str, lo: int, hi: int, default: int | None = None) -> int:
    if v is None:
        if default is None:
            raise BadRequest(f"falta {name}")
        return default
    try:
        x = int(v)
    except (TypeError, ValueError):
        raise BadRequest(f"{name} no es un número") from None
    if not lo <= x <= hi:
        raise BadRequest(f"{name} fuera de rango ({lo}-{hi})")
    return x


def decode_image(b64: str) -> Image.Image:
    """Una imagen en base64 (PNG o JPEG, con o sin el prefijo data:)."""
    s = str(b64 or "")
    if s.startswith("data:"):
        s = s.split(",", 1)[1]
    try:
        return Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")
    except Exception as e:  # noqa: BLE001
        raise BadRequest(f"imagen de referencia ilegible ({type(e).__name__})") from None


def encode_image(img: Image.Image, fmt: str = "png", quality: int = 95) -> str:
    buf = io.BytesIO()
    if fmt == "jpeg":
        img.convert("RGB").save(buf, "JPEG", quality=quality)
    else:
        img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def parse(payload: dict[str, Any]) -> dict[str, Any]:
    """La petición validada (BadRequest si no cumple el contrato)."""
    if not isinstance(payload, dict):
        raise BadRequest("el cuerpo no es un objeto JSON")
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        raise BadRequest("falta prompt")
    w = _int(payload.get("width"), "width", 64, MAX_SIDE)
    h = _int(payload.get("height"), "height", 64, MAX_SIDE)
    refs = payload.get("refs") or []
    if not isinstance(refs, list) or len(refs) > MAX_REFS:
        raise BadRequest(f"refs: como mucho {MAX_REFS} imágenes")
    fmt = str(payload.get("format") or "png").lower()
    if fmt not in ("png", "jpeg", "jpg"):
        raise BadRequest("format: png o jpeg")
    seed = payload.get("seed")
    return {"prompt": prompt[:MAX_PROMPT_CHARS], "negative": str(payload.get("negative") or "").strip(),
            "width": w, "height": h,
            "seed": _int(seed, "seed", 0, 2**32 - 1) if seed is not None else int(time.time() * 1000) % 1_000_000,
            "steps": _int(payload.get("steps"), "steps", 1, 100, DEFAULT_STEPS),
            "refs": [decode_image(r) for r in refs], "ref_strength": float(payload.get("ref_strength") or 1.0),
            "n": _int(payload.get("n"), "n", 1, MAX_N, 1), "format": "jpeg" if fmt in ("jpeg", "jpg") else "png",
            "quality": _int(payload.get("quality"), "quality", 50, 100, 95), "hd": bool(payload.get("hd")),
            "edit": bool(payload.get("edit"))}


def native_size(w: int, h: int) -> tuple[int, int]:
    """El tamaño nativo de Qwen-Image-2512 con la proporción más parecida a w × h."""
    r = math.log(w / h)
    return min(NATIVE, key=lambda s: (abs(math.log(s[0] / s[1]) - r), -s[0] * s[1]))


def needs_upscale(w: int, h: int, native: tuple[int, int]) -> bool:
    return w > native[0] * UPSCALE_OVER or h > native[1] * UPSCALE_OVER


def fit(img: Image.Image, w: int, h: int) -> Image.Image:
    """Llena w × h sin deformar (recorta lo que sobra de la proporción), con Lanczos."""
    sw, sh = img.size
    if (sw, sh) == (w, h):
        return img
    scale = max(w / sw, h / sh)
    nw, nh = max(w, int(round(sw * scale))), max(h, int(round(sh * scale)))
    img = img.resize((nw, nh), Image.LANCZOS)
    x, y = (nw - w) // 2, (nh - h) // 2
    return img.crop((x, y, x + w, y + h))


def _hd(img: Image.Image) -> Image.Image:
    if max(img.size) > HD_SIDE:
        img = img.copy()
        img.thumbnail((HD_SIDE, HD_SIDE), Image.LANCZOS)
    return img


# ---------- modo de prueba ----------

def _fake_image(req: dict[str, Any], i: int, size: tuple[int, int]) -> Image.Image:
    """Una imagen sintética con textura (no lisa: el control de bandas negras y de nitidez la miran igual que una de
    verdad), distinta por prompt y semilla; con fichas, lleva un trozo de la primera."""
    import numpy as np
    w, h = size
    hx = hashlib.md5(f"{req['prompt']}|{req['seed'] + i}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(hx[:8], "little"))
    base = np.array([hx[0], hx[1], hx[2]], np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    grad = (xx / max(1, w - 1))[..., None] * 90 + (yy / max(1, h - 1))[..., None] * 60
    noise = rng.normal(0, 18, (h, w, 1)).astype(np.float32)
    arr = np.clip(base * 0.55 + 30 + grad + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(np.repeat(arr, 3, axis=2) if arr.shape[2] == 1 else arr, "RGB")
    d = ImageDraw.Draw(img)
    for k in range(6):
        x0, y0 = int(rng.integers(0, max(1, w - 40))), int(rng.integers(0, max(1, h - 40)))
        d.ellipse((x0, y0, x0 + w // 6, y0 + h // 6), fill=(int(hx[3 + k]), int(hx[9 + k]), int(hx[4 + k])))
    if req["refs"]:
        ref = req["refs"][0].copy()
        ref.thumbnail((w // 4, h // 4))
        img.paste(ref, (w - ref.width - 8, 8))
    return img


# ---------- la GPU ----------

def _vram():
    """El ayudante del servidor (nucleo: vram.ensure(name)), donde esté; None si no hay (modo de prueba)."""
    for name in ([f"{__package__.rsplit('.', 1)[0]}.vram"] if __package__ and "." in __package__ else []) + ["vram"]:
        try:
            return importlib.import_module(name)
        except Exception:  # noqa: BLE001
            continue
    return None


def _ensure_vram(name: str) -> None:
    v = _vram()
    if v is None:
        return
    reg = getattr(v, "register", None)
    if callable(reg) and not _S.get(f"reg_{name}"):
        try:   # (si el ayudante sabe liberar los modelos de cada parte, que sepa liberar este)
            reg(name, unload=unload, gb=VRAM_GB.get(name, 0.0))
            _S[f"reg_{name}"] = True
        except Exception:  # noqa: BLE001
            pass
    v.ensure(name)


def unload() -> None:
    """Libera la VRAM de las imágenes (lo llama vram.ensure cuando otra parte necesita la GPU). Sin el cerrojo: el
    servidor lo llama desde otra petición y la nuestra espera en vram.ensure antes de usar la GPU."""
    for k in ("t2i", "t2i_tr", "edit", "edit_tr", "up"):
        _S[k] = None
    _S["lora"] = {}
    try:
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _free_gb() -> float:
    import torch
    free, _total = torch.cuda.mem_get_info()
    return free / 1024 ** 3


def _ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / 1024 ** 3
    except Exception:  # noqa: BLE001
        return 0.0


def _on_gpu(module) -> bool:
    return module is not None and next(module.parameters()).device.type == "cuda"


def _park(kind: str) -> None:
    """El transformer de `kind` ("t2i" o "edit") fuera de la GPU para que quepa el otro (los dos en bf16 son 82 GB):
    a la RAM si cabe (volver cuesta segundos) o, si no, soltado (se vuelve a leer del disco)."""
    import torch
    key = "t2i_tr" if kind == "t2i" else "edit_tr"
    tr = _S.get(key)
    if not _on_gpu(tr):
        return
    if _ram_gb() > VRAM_GB[EDIT_NAME] + 16:
        tr.to("cpu")
    else:
        _S[key] = None
        pipe = _S.get(kind)
        if pipe is not None:
            pipe.transformer = None
        del tr
    torch.cuda.empty_cache()


def _transformer(kind: str):
    """El transformer de `kind`, en la GPU (cargándolo o trayéndolo de la RAM; el otro sale si no caben los dos)."""
    import torch
    from diffusers import QwenImageTransformer2DModel
    key, other = ("t2i_tr", "edit") if kind == "t2i" else ("edit_tr", "t2i")
    tr = _S.get(key)
    if _on_gpu(tr):
        return tr
    if _free_gb() < VRAM_GB[EDIT_NAME] + 6:
        _park(other)
    if tr is None:
        tr = QwenImageTransformer2DModel.from_pretrained(T2I_REPO if kind == "t2i" else EDIT_REPO,
                                                         subfolder="transformer", torch_dtype=torch.bfloat16)
        _S[key] = tr
        _S["lora"].pop(kind, None)
    tr.to("cuda")
    return tr


def _t2i():
    """Qwen-Image-2512 entero (su codificador de texto y su VAE se quedan en la GPU: los usa también el de edición)."""
    import torch
    from diffusers import QwenImagePipeline
    tr = _transformer("t2i")
    pipe = _S.get("t2i")
    if pipe is None:
        pipe = QwenImagePipeline.from_pretrained(T2I_REPO, transformer=tr, torch_dtype=torch.bfloat16).to("cuda")
        pipe._cine_sched = pipe.scheduler
        _S["t2i"] = pipe
    pipe.transformer = tr
    return pipe


def _edit():
    """Qwen-Image-Edit-2511 con el codificador de texto y el VAE de la 2512 (son los mismos: solo cambia el
    transformer)."""
    import torch
    from diffusers import QwenImageEditPlusPipeline
    base = _S.get("t2i") or _t2i()
    tr = _transformer("edit")
    pipe = _S.get("edit")
    if pipe is None:
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            EDIT_REPO, transformer=tr, text_encoder=base.text_encoder, vae=base.vae, tokenizer=base.tokenizer,
            torch_dtype=torch.bfloat16)
        pipe._cine_sched = pipe.scheduler
        _S["edit"] = pipe
    pipe.transformer = tr
    return pipe


def _lightning(pipe, kind: str, on: bool) -> None:
    """El LoRA de 8 pasos (y su planificador) puesto o quitado."""
    from diffusers import FlowMatchEulerDiscreteScheduler
    repo, file = T2I_LORA if kind == "t2i" else EDIT_LORA
    if on:
        if not _S["lora"].get(kind):
            pipe.load_lora_weights(repo, weight_name=file, adapter_name="lightning")
            _S["lora"][kind] = True
        if hasattr(pipe, "enable_lora"):
            pipe.enable_lora()
        pipe.set_adapters(["lightning"], [1.0])
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(LIGHTNING_SCHEDULER)
    else:
        if _S["lora"].get(kind):
            pipe.disable_lora()
        pipe.scheduler = getattr(pipe, "_cine_sched", pipe.scheduler)


def _ref_area(pipe, area: int) -> None:
    """El tamaño al que el modelo de edición lee cada imagen de entrada (VAE_IMAGE_SIZE del módulo de diffusers)."""
    import sys
    mod = sys.modules.get(type(pipe).__module__)
    if mod is not None and hasattr(mod, "VAE_IMAGE_SIZE"):
        mod.VAE_IMAGE_SIZE = int(area)


def _call(pipe, **kw):
    """La llamada al pipeline con solo los argumentos que admite (cambian entre versiones de diffusers)."""
    import inspect
    params = inspect.signature(pipe.__call__).parameters
    if not any(p.kind == p.VAR_KEYWORD for p in params.values()):
        kw = {k: v for k, v in kw.items() if k in params}
    return pipe(**kw)


def _upscaler():
    """Real-ESRGAN x2plus (spandrel lo carga sin el paquete basicsr), en fp16."""
    if _S["up"] is None:
        from spandrel import ModelLoader
        path = _weights_file(UPSCALER_NAME, UPSCALER_FILE, UPSCALER_URL)
        _S["up"] = ModelLoader().load_from_file(path).to("cuda").eval().half()
    return _S["up"]


def _weights_file(name: str, file: str, url: str) -> str:
    """El archivo de pesos: el que bajó models.py (nucleo) o, si no, bajado aquí a la caché."""
    for mod in (["models"] + ([f"{__package__.rsplit('.', 1)[0]}.models"] if __package__ and "." in __package__ else [])):
        try:
            m = importlib.import_module(mod)
            p = getattr(m, "path_for", None) or getattr(m, "path", None)
            if callable(p):
                got = p(name)
                if got and os.path.exists(str(got)):
                    return str(got)
        except Exception:  # noqa: BLE001
            continue
    cache = os.path.join(os.environ.get("CINESTORIA_WEIGHTS") or os.environ.get("HF_HOME") or "/workspace/weights",
                         "cinestoria")
    os.makedirs(cache, exist_ok=True)
    dst = os.path.join(cache, file)
    if not os.path.exists(dst):
        import urllib.request
        urllib.request.urlretrieve(url, dst + ".part")
        os.replace(dst + ".part", dst)
    return dst


def upscale2(img: Image.Image) -> Image.Image:
    import numpy as np
    import torch
    model = _upscaler()
    x = torch.from_numpy(np.asarray(img.convert("RGB"), np.float32) / 255.0).permute(2, 0, 1)[None]
    with torch.inference_mode():
        y = model(x.to("cuda").half()).float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((y * 255.0 + 0.5).astype("uint8"), "RGB")


def _generate(req: dict[str, Any], size: tuple[int, int]) -> tuple[list[Image.Image], str]:
    """Las imágenes en la GPU, a su tamaño nativo. Devuelve (imágenes, modelo)."""
    import torch
    fast = req["steps"] <= FAST_STEPS
    neg = (req["negative"] + " " + NEGATIVE_BASE).strip() if req["negative"] else NEGATIVE_BASE
    gens = [torch.Generator(device="cuda").manual_seed(req["seed"] + i) for i in range(req["n"])]
    common = {"prompt": req["prompt"], "negative_prompt": neg, "true_cfg_scale": 1.0 if fast else TRUE_CFG,
              "num_inference_steps": req["steps"], "num_images_per_prompt": req["n"], "generator": gens,
              "width": size[0], "height": size[1], "max_sequence_length": MAX_SEQ}
    if req["refs"]:
        _ensure_vram(EDIT_NAME)
        pipe = _edit()
        _lightning(pipe, "edit", fast)
        _ref_area(pipe, size[0] * size[1] if req["edit"] else EDIT_REF_AREA)   # (la imagen que se edita, entera)
        out = _call(pipe, image=req["refs"], guidance_scale=1.0, **common).images
        name = EDIT_NAME
    else:
        _ensure_vram(T2I_NAME)
        pipe = _t2i()
        _lightning(pipe, "t2i", fast)
        out = _call(pipe, **common).images
        name = T2I_NAME
    return list(out), name + ("+lightning-8" if fast else "")


def handle(payload: dict[str, Any]) -> dict[str, Any]:
    """POST /image. BadRequest si la petición no cumple el contrato; cualquier otro error, de la GPU (el servidor lo
    devuelve como 500 y Cinestoria pasa esa imagen a su proveedor de reserva)."""
    req = parse(payload)
    w, h = req["width"], req["height"]
    native = native_size(w, h)
    up = needs_upscale(w, h, native)
    t0 = time.perf_counter()
    with _LOCK:
        if fake():
            raw = [_fake_image(req, i, native) for i in range(req["n"])]
            model = ("fake:" + (EDIT_NAME if req["refs"] else T2I_NAME)
                     + ("+lightning-8" if req["steps"] <= FAST_STEPS else ""))
            big = [im.resize((im.width * 2, im.height * 2), Image.BICUBIC) for im in raw] if (up or req["hd"]) else []
        else:
            raw, model = _generate(req, native)
            big = [upscale2(im) for im in raw] if (up or req["hd"]) else []
        outs = [fit(big[i] if up else raw[i], w, h) for i in range(len(raw))]
        seconds = time.perf_counter() - t0
    res: dict[str, Any] = {"images": [encode_image(im, req["format"], req["quality"]) for im in outs],
                           "seconds": round(seconds, 3), "model": model, "native": list(native), "upscaled": up,
                           "steps": req["steps"]}
    if req["hd"]:
        res["images_hd"] = [encode_image(_hd(im), "jpeg", 88) for im in big]
    if fake():
        res["fake"] = True
    return res


def loaded() -> list[str]:
    """Los modelos de imagen que están en la GPU ahora (para GET /health)."""
    if fake():
        return [f"fake:{T2I_NAME}"]
    out = []
    try:
        if _on_gpu(_S.get("t2i_tr")):
            out.append(T2I_NAME)
        if _on_gpu(_S.get("edit_tr")):
            out.append(EDIT_NAME)
    except Exception:  # noqa: BLE001
        pass
    if _S.get("up") is not None:
        out.append(UPSCALER_NAME)
    return out


def warmup() -> None:
    """Carga Qwen-Image-2512 y Real-ESRGAN antes de la primera petición (lo llama server.py al arrancar, en otro hilo):
    cargar 58 GB tarda más de lo que el proxy de RunPod espera a una petición (100 s)."""
    if fake():
        return
    with _LOCK:
        _ensure_vram(T2I_NAME)
        _t2i()
        _upscaler()


# ---------- la ruta (la monta server.py de nucleo) ----------

ROUTES = [("POST", "/image", handle)]

try:
    from fastapi import APIRouter, Header, HTTPException, Request
    from fastapi.concurrency import run_in_threadpool

    router = APIRouter()

    def _auth(authorization: str | None) -> None:
        """La misma comprobación del token que hace el servidor (por si este router se monta sin ella)."""
        token = os.environ.get("CINESTORIA_TOKEN", "")
        if token and (authorization or "") != f"Bearer {token}":
            raise HTTPException(status_code=401, detail="token")

    @router.post("/image")
    async def image_route(request: Request, authorization: str | None = Header(default=None)):
        _auth(authorization)
        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001
            raise HTTPException(status_code=422, detail="el cuerpo no es JSON") from None
        try:
            return await run_in_threadpool(handle, payload)
        except BadRequest as e:
            raise HTTPException(status_code=422, detail=str(e)) from None
except ImportError:   # (sin FastAPI: solo handle(), para probarlo)
    router = None
