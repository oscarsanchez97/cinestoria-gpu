# 1.30: el worker de la GPU de Cinestoria («Tu GPU», en un pod de RunPod de la cuenta del cliente).
# Imagen pública en GHCR: la construye .github/workflows/gpu-image.yml (ver NOTAS de la 1.30).
#
#   docker build -t cinestoria-gpu docker/gpu
#   docker run --rm --gpus all -p 8000:8000 -e CINESTORIA_TOKEN=un-token-largo cinestoria-gpu
#
# - Python 3.12, torch 2.14.1 de PyPI (trae CUDA 13: el pod pide allowedCudaVersions 13.0). Para máquinas con un driver
#   de CUDA 12.8: --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128 (y NARRA_GPU_CUDA=12.8,12.9,13.0).
# - Un solo entorno para imágenes (Qwen-Image), clips (Wan 2.2), música (ACE-Step) y voz (Chatterbox), con las versiones
#   de constraints.txt (pip no puede bajar torch ni diffusers a escondidas). chatterbox-tts 0.1.7 fija torch 2.6.0,
#   diffusers 0.29 y numpy<2, que romperían lo demás: se instala sin dependencias y con las suyas de requirements-tts.txt
#   (la acción de GitHub comprueba que se importa). qwen-tts 0.1.1 (diseñar voces) pide transformers 4.57.3: va en su
#   entorno aparte, /opt/qwen (CINESTORIA_QWEN_PY), que ve el torch del sistema.
# - ACE-Step 1.5 (música) fija torch 2.10.0+cu128 y transformers < 4.58 y solo se instala con uv (su nano-vllm va
#   dentro del repositorio): va en su propio entorno, /opt/acestep/.venv, hecho con «uv sync» en el commit ACESTEP_REF,
#   y el servidor le pasa las peticiones de /music a otro proceso en ese Python (CINESTORIA_PY_MUSIC; ver server.py).
#   Fíjalo a un commit probado: --build-arg ACESTEP_REF=<sha>.
# - Practical-RIFE (MIT, la interpolación de los clips; opcional: sin él, minterpolate de ffmpeg) en /opt/Practical-RIFE.
# - ffmpeg de Debian (con libx264: los clips y la voz); la acción de GitHub lo comprueba.
# - Unos 16 GB de imagen (dos torch: el del sistema y el de ACE-Step).
# - Sin pesos dentro: se bajan de Hugging Face al arrancar el pod (o están en el volumen de red), ver models.py.
FROM python:3.12-slim-bookworm

ARG TORCH_INDEX=""
ARG ACESTEP_REF=main
ARG RIFE_REF=main
ARG UV_VERSION=0.8.0
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_XET_HIGH_PERFORMANCE=1 \
    CINESTORIA_WEIGHTS=/workspace/cinestoria/pesos \
    CINESTORIA_MODELS=/workspace/cinestoria/pesos \
    CINESTORIA_CACHE=/workspace/cinestoria \
    HF_HOME=/workspace/cinestoria/hf \
    CINESTORIA_QWEN_PY=/opt/qwen/bin/python \
    CINESTORIA_PY_MUSIC=/opt/acestep/.venv/bin/python \
    CINESTORIA_RIFE_DIR=/opt/Practical-RIFE

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 sox git ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY constraints.txt requirements*.txt ./
RUN if [ -n "$TORCH_INDEX" ]; then \
        pip install torch==2.14.1 torchvision==0.29.1 torchaudio==2.11.1 --index-url "$TORCH_INDEX"; \
    else \
        pip install -c constraints.txt torch torchvision torchaudio; \
    fi \
    && pip install -c constraints.txt -r requirements.txt \
    && pip install -c constraints.txt -r requirements-image.txt -r requirements-clip.txt \
    && pip install --no-deps chatterbox-tts==0.1.7 \
    && pip install -c constraints.txt -r requirements-tts.txt \
    && python -m venv --system-site-packages /opt/qwen \
    && /opt/qwen/bin/pip install qwen-tts==0.1.1

# la música: ACE-Step 1.5 en su entorno (uv sync, como dice su guía), con lo que el servidor necesita para arrancar ahí
RUN pip install "uv==${UV_VERSION}" \
    && git clone https://github.com/ace-step/ACE-Step-1.5 /opt/acestep \
    && git -C /opt/acestep checkout "${ACESTEP_REF}" \
    && cd /opt/acestep && UV_PROJECT_ENVIRONMENT=/opt/acestep/.venv uv sync --no-dev \
    && uv pip install --python /opt/acestep/.venv/bin/python -r /app/requirements-music.txt \
    && rm -rf /root/.cache/uv

# la interpolación de los clips (opcional)
RUN git clone https://github.com/hzwer/Practical-RIFE /opt/Practical-RIFE \
    && git -C /opt/Practical-RIFE checkout "${RIFE_REF}" \
    && rm -rf /opt/Practical-RIFE/.git

COPY . .

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "75"]
