"""1.30 (voz): diseña la voz de referencia de una voz de serie de Cinestoria con Qwen3-TTS-12Hz-1.7B-VoiceDesign
(Apache-2.0) a partir de su descripción (docker/gpu/handlers/voces.json). Lo lanza tts.design_reference en un proceso
aparte, con el Python del entorno de qwen-tts (CINESTORIA_QWEN_PY): qwen-tts 0.1.1 pide transformers 4.57.3 y
chatterbox-tts 0.1.7, transformers 5.2.0, así que no pueden ir en el mismo entorno.

    python disenar_voz.py --instruct "Male voice: …" --texto "En el invierno…" --idioma Spanish --semilla 1000 \
        --salida /workspace/cinestoria/voces/_diseno_es-ES-hombre-1.wav

Escribe un WAV mono de 16 bits y una línea JSON con lo que ha hecho. Una voz así no es la grabación ni el clon de nadie.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import wave


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruct", required=True)
    ap.add_argument("--texto", required=True)
    ap.add_argument("--idioma", default="Spanish")
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--modelo", default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    a = ap.parse_args(argv)
    t0 = time.monotonic()
    import numpy as np
    import torch
    from qwen_tts import Qwen3TTSModel
    try:
        import flash_attn  # noqa: F401
        attn = "flash_attention_2"
    except ImportError:
        attn = "sdpa"
    model = Qwen3TTSModel.from_pretrained(a.modelo, device_map="cuda:0" if torch.cuda.is_available() else "cpu",
                                          dtype=torch.bfloat16, attn_implementation=attn)
    torch.manual_seed(a.semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.semilla)
    wavs, sr = model.generate_voice_design(text=a.texto, language=a.idioma, instruct=a.instruct)
    x = np.asarray(wavs[0], dtype=np.float32).squeeze()
    peak = float(np.abs(x).max()) or 1.0
    pcm = (np.clip(x / peak * 0.9, -1, 1) * 32767).astype("<i2")
    with wave.open(a.salida, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())
    print(json.dumps({"ok": True, "sr": int(sr), "segundos": round(len(x) / sr, 2),
                      "gpu_s": round(time.monotonic() - t0, 1)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
