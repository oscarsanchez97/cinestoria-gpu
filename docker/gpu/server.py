"""1.30: el worker de la GPU de Cinestoria (dentro del pod de RunPod): una API JSON con los modelos abiertos.

    GET  /health             {"ok", "gpu", "vram_gb", "loaded", "version", …}
    POST /image /tts /align /music /clip   (las de los handlers: ver handlers/__init__.py y el contrato 1.30)
    POST /jobs               {"path": "/clip", "payload": {...}} → 202 {"id", "job"}: lo mismo, en segundo plano
    GET  /jobs/{id}          {"state": queued|preparing|running|done|error, "result"? , "error"?, "code"?}

- Todas las rutas piden «Authorization: Bearer <CINESTORIA_TOKEN>» (el token de ese pod, que pone Cinestoria al
  crearlo). Sin token configurado, no responde a nadie.
- El proxy de RunPod corta las conexiones a los 100 s: con «Prefer: respond-async» (o por POST /jobs), la petición se
  encola, se responde 202 {"job": id} al momento y el resultado se consulta en /jobs/{id}. El resultado se guarda hasta
  que se recoge (y 15 min más; uno que nadie recoge, 1 h). Sin esa cabecera, se responde al terminar (llamadas cortas).
- Cada handler se monta de forma genérica: sus ROUTES (un dict {ruta: función} o una lista [("POST", ruta,
  función)]; si no las tiene, su `router` de FastAPI), sus MODELS (licencia comercial comprobada al montar: si no, el
  handler no se monta y /health lo dice), su VRAM_GB (un número o {nombre: GB}) con su unload() (vram.register), su
  warmup() y loaded() y sus NOTES.
- Antes de su primera petición, se bajan sus pesos si faltan (models.ensure) y se carga (warmup) en el estado
  «preparing» (no cuenta en el tiempo máximo de la llamada); cada handler tiene su turno en la GPU (CONCURRENCY, 1).
- CINESTORIA_FAKE=1: salidas sintéticas sin torch (las pruebas del contrato en una CPU).
- Un handler con dependencias que chocan con las de los demás (ACE-Step pide torch 2.10 y transformers < 4.58;
  Chatterbox, transformers 5.2) va en su propio Python: CINESTORIA_PY_<NOMBRE> (p. ej. CINESTORIA_PY_MUSIC=
  /opt/acestep/.venv/bin/python). Este servidor arranca otro igual en ese Python, solo con ese handler, en
  127.0.0.1 y con un token suyo, y le pasa sus peticiones; la memoria de la GPU se sigue repartiendo aquí (vram.ensure
  antes de cada petición; para soltarla, POST /unload al otro). El módulo del handler tiene que importarse aquí sin sus
  dependencias (lo pesado, dentro de las funciones: el contrato de handlers/__init__.py). CINESTORIA_HANDLERS limita
  los handlers que se cargan (los demás, fuera).

Se arranca con: uvicorn server:app --host 0.0.0.0 --port 8000 (lo hace el Dockerfile).
"""
from __future__ import annotations

import atexit
import hmac
import importlib
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
import types
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

import models
import vram
from handlers import NAMES

VERSION = "1.30"
KEEP_S = 900.0        # lo que se guarda un resultado ya recogido
KEEP_UNFETCHED_S = 3600.0
MAX_KEPT = 500


def handler_names() -> tuple[str, ...]:
    only = [x.strip() for x in (os.environ.get("CINESTORIA_HANDLERS") or "").split(",") if x.strip()]
    return tuple(n for n in NAMES if not only or n in only)


def load_handlers() -> tuple[dict, dict]:
    """Los handlers que cargan y tienen licencias comerciales (y el error de los que no: el resto funciona igual).
    Los que tienen su propio Python (CINESTORIA_PY_<NOMBRE>) se cargan como «Child»: el mismo módulo, en otro proceso."""
    mods, errors = {}, {}
    for n in handler_names():
        try:
            m = importlib.import_module(f"handlers.{n}")
            for s in getattr(m, "MODELS", []) or []:
                models.check_license(s)
            py = child_python(n)
            mods[n] = Child(n, m, py).module() if py else m
        except Exception as e:  # noqa: BLE001
            errors[n] = f"{type(e).__name__}: {e}"[:200]
    return mods, errors


def child_python(name: str) -> str:
    """El Python propio de un handler, si lo tiene y existe (y no somos ya ese proceso)."""
    if os.environ.get("CINESTORIA_CHILD") == "1":
        return ""
    py = (os.environ.get(f"CINESTORIA_PY_{name.upper()}") or "").strip()
    if not py or not Path(py).exists():
        return ""
    try:
        if Path(py).resolve() == Path(sys.executable).resolve():
            return ""
    except OSError:
        pass
    return py


class ChildError(RuntimeError):
    pass


class Child:
    """Un handler en su propio Python: otro server.py (solo con ese handler) en 127.0.0.1, con un token suyo.
    Arranca con su primera petición (o al arrancar el pod, si está en CINESTORIA_PRELOAD) y se rearranca si se cae."""
    BOOT_S = float(os.environ.get("CINESTORIA_CHILD_BOOT_S") or 300)
    CALL_S = float(os.environ.get("CINESTORIA_CHILD_CALL_S") or 3600)

    def __init__(self, name: str, mod, py: str):
        self.name, self.mod, self.py = name, mod, py
        self.token = secrets.token_hex(24)
        self.port = 0
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()
        self._loaded: tuple[float, list[str]] = (0.0, [])
        atexit.register(self.stop)

    def _url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"}

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self) -> None:
        import requests  # noqa: PLC0415
        with self.lock:
            if self.alive():
                return
            with socket.socket() as sk:
                sk.bind(("127.0.0.1", 0))
                self.port = sk.getsockname()[1]
            env = {**os.environ, "CINESTORIA_CHILD": "1", "CINESTORIA_HANDLERS": self.name,
                   "CINESTORIA_TOKEN": self.token, "CINESTORIA_NO_PRELOAD": "1", "PYTHONUNBUFFERED": "1"}
            self.proc = subprocess.Popen(
                [self.py, "-m", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", str(self.port),
                 "--log-level", "warning"], cwd=str(Path(__file__).resolve().parent), env=env,
                stdin=subprocess.DEVNULL)
            end = time.time() + self.BOOT_S
            while time.time() < end:
                if self.proc.poll() is not None:
                    raise ChildError(f"el proceso de {self.name} no arranca (salió con {self.proc.returncode})")
                try:
                    if requests.get(self._url("/health"), headers=self._headers(), timeout=5).status_code == 200:
                        return
                except requests.RequestException:
                    pass
                time.sleep(0.5)
            self.stop()
            raise ChildError(f"el proceso de {self.name} no arranca (más de {self.BOOT_S:.0f} s)")

    def stop(self) -> None:
        p, self.proc = self.proc, None
        if p is not None and p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()

    def post(self, path: str, payload: dict, timeout: float | None = None) -> dict:
        import requests  # noqa: PLC0415
        if not self.alive():
            self.start()
        try:
            r = requests.post(self._url(path), json=payload, headers=self._headers(), timeout=timeout or self.CALL_S)
        except requests.RequestException as e:
            if not self.alive():
                self.proc = None
            raise ChildError(f"el proceso de {self.name} no responde ({type(e).__name__})") from None
        try:
            body = r.json()
        except ValueError:
            body = {}
        if r.status_code == 200 and isinstance(body, dict):
            return body
        msg = str((body or {}).get("error") or f"respondió {r.status_code}")[:300] if isinstance(body, dict) else ""
        if r.status_code in (400, 404, 422):
            raise ValueError(msg)
        if r.status_code == 501:
            raise NotImplementedError(msg)
        if r.status_code == 507:
            raise MemoryError(msg)
        raise ChildError(msg)

    def unload(self) -> None:
        if self.alive():
            try:
                self.post("/unload", {}, timeout=120)
            except Exception:  # noqa: BLE001
                pass

    def warmup(self) -> None:
        self.post("/warmup", {}, timeout=self.CALL_S)
        self._loaded = (0.0, [])

    def loaded(self) -> list[str]:
        if not self.alive():
            return []
        t, got = self._loaded
        if time.time() - t < 5:
            return got
        import requests  # noqa: PLC0415
        try:
            got = list(requests.get(self._url("/health"), headers=self._headers(), timeout=3).json().get("loaded") or [])
        except Exception:  # noqa: BLE001
            got = []
        self._loaded = (time.time(), got)
        return got

    def module(self):
        """Lo que ve este servidor: el mismo handler (sus rutas, pesos, memoria y notas), con todo hacia el otro proceso."""
        m, name = self.mod, str(getattr(self.mod, "NAME", self.name))
        gb = getattr(m, "VRAM_GB", None)
        keys_ = list(gb) if isinstance(gb, dict) else [name]

        def room() -> None:
            for k in keys_:   # (hace sitio aquí: lo de otras etapas se suelta antes de que el otro cargue lo suyo)
                vram.ensure(k)

        def warmup() -> None:
            room()
            self.warmup()

        def forward(path):
            def fn(req: dict) -> dict:
                if not self.alive():
                    self.start()
                room()
                return self.post(path, req)
            return fn
        out = types.SimpleNamespace(
            NAME=name, MODELS=getattr(m, "MODELS", []), VRAM_GB=gb, VRAM_GB_TOP=getattr(m, "VRAM_GB_TOP", None),
            NOTES=list(getattr(m, "NOTES", []) or []) + [f"en su propio Python ({self.py})"],
            CONCURRENCY=getattr(m, "CONCURRENCY", 1), unload=self.unload,
            warmup=warmup if callable(getattr(m, "warmup", None)) else None, loaded=self.loaded,
            ROUTES=[("POST", path, forward(path)) for method, path, fn in routes_of(m) if method == "POST"],
            child=self)
        return out


def routes_of(mod) -> list[tuple[str, str, object]]:
    """[(método, ruta, función)] de un handler: sus ROUTES como dict o como lista de tuplas."""
    r = getattr(mod, "ROUTES", None) or {}
    if isinstance(r, dict):
        return [("POST", p, f) for p, f in r.items()]
    out = []
    for item in r:
        if isinstance(item, (tuple, list)) and len(item) == 3:
            out.append((str(item[0]).upper(), item[1], item[2]))
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            out.append(("POST", item[0], item[1]))
    return out


def register_vram(name: str, mod) -> None:
    """Las reservas de memoria de un handler (VRAM_GB, con VRAM_GB_TOP si la GPU llega) y cómo soltarlo (unload)."""
    gb = getattr(mod, "VRAM_GB", None)
    unload = getattr(mod, "unload", None)
    unload = unload if callable(unload) else None
    if isinstance(gb, dict):
        for k, v in gb.items():
            vram.register(k, unload=unload, gb=float(v))
    elif isinstance(gb, (int, float)):
        top = getattr(mod, "VRAM_GB_TOP", None)
        want = float(top) if isinstance(top, (int, float)) and vram.total_gb() >= float(top) else float(gb)
        vram.register(str(getattr(mod, "NAME", name)), unload=unload, gb=want)
    elif unload is not None:   # (sin cifra: al menos, cómo soltarlo)
        vram.register(str(getattr(mod, "NAME", name)), unload=unload)


def _module_loaded(mod) -> list[str]:
    f = getattr(mod, "loaded", None)
    try:
        return [str(x) for x in (f() if callable(f) else [])]
    except Exception:  # noqa: BLE001
        return []


def create_app(token: str | None = None, fake: bool | None = None, preload: bool = True) -> FastAPI:
    tok = os.environ.get("CINESTORIA_TOKEN", "") if token is None else token
    if fake is not None:
        os.environ["CINESTORIA_FAKE"] = "1" if fake else "0"
    app = FastAPI(title="Cinestoria GPU", docs_url=None, redoc_url=None, openapi_url=None)
    mods, errors = load_handlers()
    sems = {n: threading.BoundedSemaphore(max(1, int(getattr(m, "CONCURRENCY", 1) or 1))) for n, m in mods.items()}
    pool = ThreadPoolExecutor(max_workers=max(2, int(os.environ.get("CINESTORIA_THREADS") or 8)), thread_name_prefix="gpu")
    jobs: dict[str, dict] = {}
    lock = threading.Lock()
    activity = {"t": time.time()}
    table: dict[str, tuple[str, object]] = {}   # ruta → (handler, función)

    def authed(request: Request) -> bool:
        h = request.headers.get("authorization") or ""
        given = h[7:].strip() if h.lower().startswith("bearer ") else ""
        return bool(tok) and hmac.compare_digest(given.encode(), tok.encode())

    @app.middleware("http")
    async def auth(request: Request, call_next):
        if not authed(request):
            return JSONResponse({"error": "sin autorización"}, status_code=401)
        return await call_next(request)

    def busy() -> int:
        with lock:
            return sum(1 for j in jobs.values() if j["state"] in ("queued", "preparing", "running"))

    def tidy() -> None:
        now = time.time()
        with lock:
            old = [k for k, j in jobs.items() if j.get("done") and now - j["done"] > (
                KEEP_S if j.get("fetched") else KEEP_UNFETCHED_S)]
            for k in old:
                jobs.pop(k, None)
            if len(jobs) > MAX_KEPT:
                for k in sorted((k for k, j in jobs.items() if j.get("done")), key=lambda k: jobs[k]["done"])[:len(jobs) - MAX_KEPT]:
                    jobs.pop(k, None)

    def prepare(mod) -> None:
        """Sus pesos y, si aún no tiene nada cargado, su warmup (fuera del turno de la GPU y del tiempo de la llamada)."""
        models.ensure_all(getattr(mod, "MODELS", []) or [])
        w = getattr(mod, "warmup", None)
        if callable(w) and not _module_loaded(mod):
            w()

    def run(job: dict, name: str, fn, req: dict) -> None:
        mod = mods[name]
        try:
            job["state"] = "preparing"
            prepare(mod)
            job["state"] = "queued"
            with sems[name]:
                job["state"] = "running"
                job["started"] = time.time()
                out = fn(req)
            if not isinstance(out, dict):
                raise TypeError("el handler no ha devuelto un diccionario")
            job.update(state="done", result=out)
        except ValueError as e:   # (también los BadRequest de los handlers, que lo son)
            job.update(state="error", error=str(e)[:300], code=400)
        except NotImplementedError as e:
            job.update(state="error", error=str(e)[:300], code=501)
        except ImportError as e:   # 1.30: falta la librería del modelo en este pod: como un handler sin modelo (501)
            job.update(state="error", error=f"falta en este pod: {e}"[:300], code=501)
        except MemoryError as e:
            job.update(state="error", error=f"sin memoria en la GPU: {e}"[:300], code=507)
        except Exception as e:  # noqa: BLE001
            code = 400 if type(e).__name__ in ("BadRequest", "HTTPException") and getattr(e, "status_code", 400) < 500 else 500
            job.update(state="error", error=f"{type(e).__name__}: {getattr(e, 'detail', None) or e}"[:300], code=code)
        finally:
            job["done"] = time.time()
            activity["t"] = time.time()

    def submit(path: str, req: dict, background: bool):
        name, fn = table[path]
        tidy()
        activity["t"] = time.time()
        jid = uuid.uuid4().hex
        job = {"state": "queued", "path": path, "t": time.time()}
        with lock:
            jobs[jid] = job
        if background:
            pool.submit(run, job, name, fn, req)
        return jid, job

    def mount(name: str, path: str, fn) -> None:
        table[path] = (name, fn)

        async def endpoint(request: Request):
            try:
                req = await request.json()
            except Exception:  # noqa: BLE001
                return JSONResponse({"error": "el cuerpo no es JSON"}, status_code=400)
            if not isinstance(req, dict):
                return JSONResponse({"error": "el cuerpo tiene que ser un objeto JSON"}, status_code=400)
            if "respond-async" in (request.headers.get("prefer") or "").lower():
                jid, _ = submit(path, req, True)
                return JSONResponse({"job": jid, "id": jid, "state": "queued"}, status_code=202)
            jid, job = submit(path, req, False)
            await run_in_threadpool(run, job, name, fn, req)
            with lock:
                jobs.pop(jid, None)
            if job["state"] == "done":
                return JSONResponse(job["result"])
            return JSONResponse({"error": job.get("error") or "error"}, status_code=int(job.get("code") or 500))
        endpoint.__name__ = f"{name}_{path.strip('/').replace('/', '_')}"
        app.add_api_route(path, endpoint, methods=["POST"])

    for name, mod in mods.items():
        register_vram(name, mod)
        rts = routes_of(mod)
        for method, path, fn in rts:
            if method == "POST" and callable(fn):
                mount(name, path, fn)
        if not rts and getattr(mod, "router", None) is not None:   # (sin ROUTES: su router tal cual, sin segundo plano)
            app.include_router(mod.router)

    @app.get("/health")
    def health():
        n = busy()
        loaded = list(dict.fromkeys(vram.loaded() + [x for m in mods.values() for x in _module_loaded(m)]))
        notes = {k: [str(x) for x in getattr(m, "NOTES", [])][:20] for k, m in mods.items() if getattr(m, "NOTES", None)}
        return {"ok": True, "gpu": vram.gpu_name(), "vram_gb": vram.total_gb(), "loaded": loaded,
                "version": VERSION, "fake": os.environ.get("CINESTORIA_FAKE") == "1",
                "idle_s": 0.0 if n else round(time.time() - activity["t"], 1), "jobs": n,
                "weights": models.status(), "handlers": {**{k: "ok" for k in mods}, **errors},
                "routes": sorted(table), "notes": notes}

    @app.post("/unload")
    def unload_all():
        """Suelta todo lo cargado en la GPU (lo pide el servidor principal a uno de sus procesos de handler)."""
        vram.release_all()
        for m in mods.values():
            u = getattr(m, "unload", None)
            if callable(u):
                try:
                    u()
                except Exception:  # noqa: BLE001
                    pass
        return {"ok": True, "loaded": vram.loaded()}

    @app.post("/warmup")
    def warmup_all():
        """Carga los handlers que tengan warmup() y aún no tengan nada cargado (con sus pesos)."""
        done = []
        for n, m in mods.items():
            w = getattr(m, "warmup", None)
            if callable(w) and not _module_loaded(m):
                models.ensure_all(getattr(m, "MODELS", []) or [])
                w()
                done.append(n)
        return {"ok": True, "warmed": done}

    @app.post("/jobs")
    async def new_job(request: Request):
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return JSONResponse({"error": "el cuerpo no es JSON"}, status_code=400)
        path = (body or {}).get("path") if isinstance(body, dict) else None
        payload = (body or {}).get("payload") if isinstance(body, dict) else None
        if path not in table or not isinstance(payload, dict):
            return JSONResponse({"error": f"ruta desconocida o sin payload: {path}"}, status_code=400)
        jid, _ = submit(path, payload, True)
        return JSONResponse({"id": jid, "job": jid, "state": "queued"}, status_code=202)

    @app.get("/jobs/{jid}")
    def job_state(jid: str):
        with lock:
            job = jobs.get(jid)
        if not job:
            return JSONResponse({"error": "no existe (o ya caducó)"}, status_code=404)
        out = {"state": job["state"]}
        if job["state"] == "done":
            out["result"] = job["result"]
            job["fetched"] = True
        elif job["state"] == "error":
            out.update(error=job.get("error"), code=job.get("code"))
            job["fetched"] = True
        return out

    if preload:
        models.preload(mods)
        want = [h.strip() for h in (os.environ.get("CINESTORIA_PRELOAD") or "").split(",") if h.strip()]
        for n, m in mods.items():   # (los de su propio Python, arrancados ya: importar torch tarda)
            ch = getattr(m, "child", None)
            if isinstance(ch, Child) and n in want:
                threading.Thread(target=lambda c=ch: _quiet(c.start), daemon=True, name=f"proceso-{n}").start()
    app.state.jobs = jobs
    app.state.handlers = mods
    return app


def _quiet(fn) -> None:
    try:
        fn()
    except Exception:  # noqa: BLE001  (se vuelve a intentar con su primera petición)
        pass


app = create_app(preload=os.environ.get("CINESTORIA_NO_PRELOAD") != "1")   # (sin CINESTORIA_TOKEN, no responde a nadie)
