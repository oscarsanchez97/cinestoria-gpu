"""1.30: los handlers del worker de la GPU, uno por área (contrato 1.30):

    image.py  (imagen)  POST /image
    tts.py    (voz)     POST /tts, POST /align
    music.py  (media)   POST /music
    clip.py   (media)   POST /clip

Cada módulo define:
    MODELS = [ {...}, ]                 # sus pesos (ver models.py: repo, revisión, archivos, GB, licencia comercial)
    ROUTES = {"/image": generate}       # ruta → función(petición: dict) -> dict (la respuesta del contrato)
    CONCURRENCY = 1                     # (opcional) cuántas peticiones suyas a la vez en la GPU

La función recibe el JSON ya leído y devuelve el JSON de la respuesta (binarios en base64). Para datos que no valen,
`raise ValueError("…")` (la petición acaba en error con ese texto). Las importaciones pesadas (torch, diffusers…) solo
dentro de las funciones. El servidor baja antes sus MODELS (models.ensure) y la memoria de la GPU se reparte con
vram.ensure(nombre, cargador, gb). Con CINESTORIA_FAKE=1 deben devolver salidas sintéticas de la forma correcta (ver
fake.py) sin torch: así se prueba el contrato en una CPU.
"""
NAMES = ("image", "tts", "music", "clip")
