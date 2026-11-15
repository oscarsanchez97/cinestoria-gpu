"""1.30: los pesos de los modelos del pod: se bajan una vez (de Hugging Face o de una URL) y se guardan en el disco del
pod (o en el volumen de red de los pesos), con reanudación y comprobación.

Cada handler declara los suyos en MODELS, una lista de diccionarios (se aceptan las formas de las tres áreas):
    {"name": "qwen-image-2512",              # nombre propio
     "repo": "Qwen/Qwen-Image-2512",          # repositorio de Hugging Face …
     "url": "https://…/x.pth", "file": "x.pth",   # … o un archivo suelto (va a CINESTORIA_MODELS/<name>/<file>)
     "revision": "<commit o rama>",           # (opcional) mejor un commit fijo: lo que se probó
     "files" | "allow" | "allow_patterns": [...],  # qué bajar del repositorio (patrones; vacío = todo)
     "size_gb" | "gb": 57.7, "vram_gb": 58,  # lo que pesa en disco y lo que ocupa en la GPU
     "license" | "licence": "Apache-2.0",     # la de su ficha: solo licencias que permiten el uso comercial
     "sha256": {"ruta/archivo": "…"},         # (opcional) comprobación de los archivos que importan
     "optional" | "lazy": True,               # (opcional) no se baja antes de tiempo: lo baja el handler si lo usa
     "prefetch": False}                       # (opcional) igual: no se baja antes de tiempo

- Los repositorios se bajan a la caché de Hugging Face (HF_HOME, en /workspace): así el `from_pretrained(repo)` de
  cada handler los encuentra sin volver a bajarlos. `path_for(nombre)` da la carpeta (o el archivo de una URL).
- huggingface_hub reanuda lo que quedó a medias y comprueba cada archivo grande con su etag; las URL se reanudan con
  «Range»; los sha256 declarados se comprueban y, si alguno no cuadra, se borra y se baja otra vez (una vez).
- Un candado en disco (O_EXCL) evita que dos pods que comparten el volumen de red bajen lo mismo a la vez.
- Nunca un modelo con licencia no comercial: `check_license` lo rechaza (y el servidor no monta ese handler).
- En modo de pruebas (CINESTORIA_FAKE=1) no se baja nada: se escriben unos archivos pequeños en su sitio.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from pathlib import Path

ALLOWED_LICENSES = {"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "cc0-1.0", "cc-by-4.0"}
BLOCKED_REPOS = {"black-forest-labs/flux.2-dev", "black-forest-labs/flux.2-klein-9b", "black-forest-labs/flux.1-dev",
                 "black-forest-labs/flux.1-kontext-dev", "black-forest-labs/flux.1-krea-dev"}
LOCK_STALE_S = 2 * 3600

_state: dict[str, str] = {}     # nombre → listo | bajando | error: …
_state_lock = threading.Lock()
_name_locks: dict[str, threading.Lock] = {}


def fake() -> bool:
    return os.environ.get("CINESTORIA_FAKE") == "1"


def root() -> Path:
    return Path(os.environ.get("CINESTORIA_WEIGHTS") or os.environ.get("CINESTORIA_MODELS") or "/workspace/cinestoria/pesos")


def norm(spec: dict) -> dict:
    """La spec con los nombres de campo de siempre (license, size_gb, files, optional)."""
    s = dict(spec)
    s["license"] = str(spec.get("license") or spec.get("licence") or "").strip()
    s["size_gb"] = float(spec.get("size_gb") or spec.get("gb") or 0.0)
    s["files"] = list(spec.get("files") or spec.get("allow") or spec.get("allow_patterns") or [])
    s["optional"] = bool(spec.get("optional") or spec.get("lazy") or spec.get("prefetch") is False)
    return s


def check_license(spec: dict) -> None:
    """Solo licencias de uso comercial (Apache 2.0, MIT, BSD, CC0, CC BY); nunca NC ni de investigación."""
    s = norm(spec)
    lic = s["license"].lower().replace(" ", "-")
    repo = str(s.get("repo") or "").strip().lower()
    if repo in BLOCKED_REPOS:
        raise ValueError(f"{s.get('name')}: {s.get('repo')} no permite el uso comercial")
    if lic not in ALLOWED_LICENSES:
        raise ValueError(f"{s.get('name')}: licencia «{lic or '?'}» no permitida (solo {sorted(ALLOWED_LICENSES)})")


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _stamp_of(spec: dict) -> str:
    s = norm(spec)
    return json.dumps({"repo": s.get("repo"), "url": s.get("url"), "revision": s.get("revision") or "main",
                       "files": sorted(s["files"]), "sha256": s.get("sha256") or {}}, sort_keys=True)


def _stamp_path(name: str) -> Path:
    return root() / ".ok" / f"{name}.json"


def path_for(name: str) -> Path | None:
    """Dónde quedaron los pesos de un modelo ya bajado (la carpeta del repositorio en la caché, o el archivo de su
    URL); None si no se ha bajado."""
    try:
        got = json.loads(_stamp_path(name).read_text(encoding="utf-8")).get("path")
    except (OSError, ValueError):
        return None
    return Path(got) if got else None


def path(name: str) -> Path:
    """La carpeta de los pesos de un modelo (la de path_for o, si no se ha bajado, la de por defecto)."""
    return path_for(name) or (root() / name)


def bad_files(spec: dict, folder: Path | None = None) -> list[str]:
    """Los archivos con sha256 declarado que faltan o no cuadran."""
    folder = Path(folder or path(spec["name"]))
    base = folder if folder.is_dir() else folder.parent
    out = []
    for rel, want in (spec.get("sha256") or {}).items():
        p = base / rel
        if not p.is_file() or _sha256(p) != str(want).lower():
            out.append(rel)
    return out


def ready(spec: dict) -> bool:
    try:
        st = json.loads(_stamp_path(spec["name"]).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if st.get("stamp") != _stamp_of(spec) or not st.get("path") or not Path(st["path"]).exists():
        return False
    return not bad_files(spec, Path(st["path"]))


class _DiskLock:
    """Candado en disco (vale en un volumen de red compartido): se crea con O_EXCL; uno de más de 2 h se da por muerto."""

    def __init__(self, p: Path):
        self.p = p

    def __enter__(self):
        self.p.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                fd = os.open(self.p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    if time.time() - self.p.stat().st_mtime > LOCK_STALE_S:
                        self.p.unlink(missing_ok=True)
                        continue
                except OSError:
                    continue
                time.sleep(2.0)

    def __exit__(self, *a):
        self.p.unlink(missing_ok=True)


def _url_file(spec: dict) -> str:
    return str(spec.get("file") or str(spec.get("url") or "").rstrip("/").rsplit("/", 1)[-1] or "pesos.bin")


def _fake_download(spec: dict) -> Path:
    """En pruebas: los archivos declarados, con un contenido pequeño y fijo, en CINESTORIA_WEIGHTS/<name>."""
    folder = root() / spec["name"]
    folder.mkdir(parents=True, exist_ok=True)
    if spec.get("url"):
        names = [_url_file(spec)]
    else:
        names = list((spec.get("sha256") or {}).keys()) or [f for f in norm(spec)["files"] if "*" not in f] \
            or ["config.json"]
    for rel in names:
        p = folder / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.write_bytes(f"fake:{spec.get('repo') or spec.get('url')}:{rel}".encode())
    return folder / names[0] if spec.get("url") else folder


def _hub_download(spec: dict) -> Path:
    """Un repositorio de Hugging Face, a su caché (HF_HOME): la carpeta de su versión."""
    from huggingface_hub import snapshot_download  # noqa: PLC0415  (solo en el pod)
    s = norm(spec)
    return Path(snapshot_download(repo_id=s["repo"], revision=s.get("revision") or None,
                                  allow_patterns=s["files"] or None,
                                  max_workers=int(os.environ.get("CINESTORIA_DL_WORKERS") or 8)))


def _url_download(spec: dict) -> Path:
    """Un archivo suelto (Real-ESRGAN de las «releases» de GitHub…) a CINESTORIA_MODELS/<name>/<file>, reanudando."""
    import requests  # noqa: PLC0415
    dest = root() / spec["name"] / _url_file(spec)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    hdr = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(spec["url"], headers=hdr, stream=True, timeout=60, allow_redirects=True) as r:
        if r.status_code not in (200, 206):
            raise IOError(f"{spec['name']}: {r.status_code} al bajar {spec['url']}")
        mode = "ab" if r.status_code == 206 and have else "wb"
        with open(part, mode) as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    part.replace(dest)
    return dest


DOWNLOAD = None   # (las pruebas ponen aquí una función propia para simular la bajada: f(spec) -> Path)


def ensure(spec: dict) -> Path:
    """Los pesos de `spec` en el disco (bajándolos si hace falta). Devuelve su carpeta (o su archivo)."""
    check_license(spec)
    name = spec["name"]
    with _state_lock:
        lk = _name_locks.setdefault(name, threading.Lock())
    with lk:
        if ready(spec):
            _set(name, "listo")
            return path(name)
        _set(name, "bajando")
        try:
            with _DiskLock(root() / ".locks" / f"{name}.lock"):
                if ready(spec):   # (otro pod lo acabó mientras esperábamos)
                    _set(name, "listo")
                    return path(name)
                fn = DOWNLOAD or (_fake_download if fake() else (_url_download if spec.get("url") else _hub_download))
                bad: list[str] = []
                for _ in range(2):
                    got = Path(fn(spec))
                    bad = bad_files(spec, got)
                    if not bad:
                        break
                    base = got if got.is_dir() else got.parent
                    for rel in bad:   # se borran y se vuelven a bajar (una vez)
                        (base / rel).unlink(missing_ok=True)
                else:
                    raise IOError(f"{name}: los archivos no cuadran con su sha256: {bad[:3]}")
                sp = _stamp_path(name)
                sp.parent.mkdir(parents=True, exist_ok=True)
                sp.write_text(json.dumps({"stamp": _stamp_of(spec), "path": str(got)}), encoding="utf-8")
        except Exception as e:
            _set(name, f"error: {str(e)[:160]}")
            raise
        _set(name, "listo")
        return got


def prefetchable(specs) -> list[dict]:
    """Las que se bajan antes de tiempo (al arrancar el pod o antes de la primera petición de su handler)."""
    return [s for s in specs or [] if (s.get("repo") or s.get("url")) and not norm(s)["optional"]]


def ensure_all(specs) -> list[Path]:
    return [ensure(s) for s in prefetchable(specs)]


def _set(name: str, v: str) -> None:
    with _state_lock:
        _state[name] = v


def status() -> dict[str, str]:
    with _state_lock:
        return dict(_state)


def preload(handlers: dict) -> threading.Thread | None:
    """Al arrancar el pod: baja en segundo plano los pesos de los handlers de CINESTORIA_PRELOAD (en ese orden, el
    del vídeo), para que estén cuando llegue su etapa (el primero, además, se carga en la GPU: su warmup())."""
    want = [h.strip() for h in (os.environ.get("CINESTORIA_PRELOAD") or "").split(",") if h.strip()]
    mods = [handlers[h] for h in want if h in handlers]
    if not mods:
        return None

    def run():
        for i, m in enumerate(mods):
            for s in prefetchable(getattr(m, "MODELS", []) or []):
                try:
                    ensure(s)
                except Exception:  # noqa: BLE001  (se reintenta al llegar su petición)
                    pass
            w = getattr(m, "warmup", None)
            if i == 0 and callable(w):
                try:
                    w()
                except Exception:  # noqa: BLE001
                    pass
    t = threading.Thread(target=run, daemon=True, name="pesos")
    t.start()
    return t


def forget(name: str) -> None:
    """Borra los pesos de un modelo bajado a CINESTORIA_WEIGHTS (para liberar disco o forzar otra bajada)."""
    shutil.rmtree(root() / name, ignore_errors=True)
    _stamp_path(name).unlink(missing_ok=True)
    _set(name, "")
