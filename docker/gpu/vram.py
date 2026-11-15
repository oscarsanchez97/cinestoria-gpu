"""1.30: la memoria de la GPU del pod (de 48 GB o de 80-96 GB si los modelos de calidad lo piden).

Cada etapa del vídeo usa sus modelos (la voz, luego las imágenes, los clips, la música). `ensure(nombre)` hace sitio
para ellos: si no caben, suelta antes los de otras partes, de los que lleven más tiempo sin usarse. Así un pod sirve a
todo el vídeo. Vale de las tres formas en que lo llaman los handlers:

    vram.ensure("qwen-image", loader=cargar_qwen, gb=40)   # carga y guarda el objeto (lo devuelve)
    vram.register("clips", unload)                         # media: cómo soltar lo suyo
    vram.register("qwen-image-2512", unload=unload, gb=58) # imagen
    vram.ensure("clips") / vram.ensure("tts", 10.0)        # una reserva: el handler carga lo suyo después

El servidor registra al montar cada handler sus VRAM_GB (un número o {nombre: GB}) con su unload(). Soltar una reserva
llama al unload() de su handler (que suelta todo lo suyo); nunca se suelta lo de la misma parte que pide sitio (cada
handler se organiza lo suyo). Las importaciones pesadas (torch) solo dentro de las funciones, y nunca en modo de
pruebas (CINESTORIA_FAKE=1): ahí solo se lleva la cuenta (con CINESTORIA_VRAM_GB, 48 por defecto)."""
from __future__ import annotations

import gc
import os
import threading
from collections import OrderedDict
from typing import Any, Callable

MARGIN_GB = 2.0
_lock = threading.RLock()
# nombre → (objeto o None si es una reserva, GB, unload o None), del menos al más reciente
_loaded: "OrderedDict[str, tuple[Any, float, Callable[[], None] | None]]" = OrderedDict()
_info: dict[str, dict[str, Any]] = {}   # nombre → {"gb", "loader", "unload"}


def fake() -> bool:
    return os.environ.get("CINESTORIA_FAKE") == "1"


def _torch():
    if fake():
        return None
    try:
        import torch  # noqa: PLC0415  (pesado: solo aquí)
    except ImportError:
        return None
    return torch if torch.cuda.is_available() else None


def gpu_name() -> str:
    t = _torch()
    if t is None:
        return "fake" if fake() else "cpu"
    try:
        return str(t.cuda.get_device_name(0))
    except Exception:  # noqa: BLE001
        return "cuda"


def total_gb() -> float:
    t = _torch()
    if t is not None:
        try:
            return round(t.cuda.mem_get_info()[1] / 1e9, 1)
        except Exception:  # noqa: BLE001
            pass
    try:
        return float(os.environ.get("CINESTORIA_VRAM_GB") or 48.0)
    except ValueError:
        return 48.0


def free_gb() -> float:
    t = _torch()
    if t is not None:
        try:
            return t.cuda.mem_get_info()[0] / 1e9
        except Exception:  # noqa: BLE001
            pass
    with _lock:
        return total_gb() - sum(gb for _, gb, _ in _loaded.values())


def register(name: str, fn: Callable | None = None, gb: float | None = None, *,
             unload: Callable[[], None] | None = None, loader: Callable[[], Any] | None = None) -> None:
    """Lo que se sabe de un modelo o de una parte: cuánta memoria pide y cómo se carga o se suelta.
    register(n, cargador, gb) · register(n, unload) · register(n, unload=f, gb=g) · register(n, loader=f, gb=g)."""
    if loader is None and unload is None and fn is not None:
        if gb is not None:
            loader = fn          # (n, cargador, gb)
        else:
            unload = fn          # (n, unload)
    with _lock:
        cur = _info.setdefault(name, {})
        if gb is not None:
            cur["gb"] = float(gb)
        if loader is not None:
            cur["loader"] = loader
        if unload is not None:
            cur["unload"] = unload


def _drop(name: str) -> None:
    obj, _, unload = _loaded.pop(name)
    del obj
    if unload is not None:
        try:
            unload()
        except Exception:  # noqa: BLE001
            pass
        for other in [k for k, v in _loaded.items() if v[2] is unload]:   # (su unload ya soltó todo lo de esa parte)
            _loaded.pop(other, None)
    gc.collect()
    t = _torch()
    if t is not None:
        try:
            t.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def ensure(name: str, loader: Callable[[], Any] | float | None = None, gb: float | None = None) -> Any:
    """Hace sitio para `name` y, con cargador, lo carga y lo devuelve (lo guarda: la segunda vez, el mismo objeto).
    Sin cargador, es una reserva: devuelve None y el handler carga lo suyo."""
    if isinstance(loader, (int, float)):   # ensure(nombre, gb)
        gb, loader = float(loader), None
    with _lock:
        if name in _loaded:
            _loaded.move_to_end(name)
            return _loaded[name][0]
        info = _info.get(name, {})
        loader = loader or info.get("loader")
        unload = info.get("unload")
        need = float(gb if gb is not None else info.get("gb") or 0.0)
        room = total_gb() - MARGIN_GB
        if loader is not None and need > room + 1e-6:
            raise MemoryError(f"{name} pide {need:.0f} GB y esta GPU tiene {total_gb():.0f} GB")
        need = min(need, room)
        while free_gb() < need + MARGIN_GB:
            victim = next((k for k, v in _loaded.items() if unload is None or v[2] is not unload), None)
            if victim is None:
                break   # (solo queda lo de la misma parte: se organiza ella)
            _drop(victim)
        obj = loader() if loader is not None else None
        _loaded[name] = (obj, need, unload)
        return obj


def release(name: str) -> None:
    with _lock:
        if name in _loaded:
            _drop(name)


def release_all() -> None:
    with _lock:
        while _loaded:
            _drop(next(iter(_loaded)))


def loaded() -> list[str]:
    with _lock:
        return list(_loaded)


def reset() -> None:
    """(Para las pruebas) olvida lo registrado y lo cargado."""
    with _lock:
        _loaded.clear()
        _info.clear()
