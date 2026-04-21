from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from pathlib import Path
import html
import os
import uuid

import numpy as np
import soundfile as sf
import librosa


app = FastAPI(title="Fluidez Lite - ComunicaLab")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".wav"}
MAX_FILE_SIZE_MB = 15
MAX_AUDIO_SECONDS = 180


# =========================
# UTILIDADES
# =========================

def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def validar_extension(filename):
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Solo archivos .wav")


async def guardar_upload(upload, destino):
    size = 0
    with open(destino, "wb") as buffer:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_FILE_SIZE_MB * 1024 * 1024:
                raise HTTPException(status_code=400, detail="Archivo demasiado grande")
            buffer.write(chunk)
    await upload.close()


# =========================
# ANÁLISIS
# =========================

def analizar_audio(path_audio):
    y, sr = sf.read(str(path_audio), always_2d=False)

    if isinstance(y, np.ndarray) and y.ndim > 1:
        y = np.mean(y, axis=1)

    y = np.asarray(y, dtype=np.float32)

    if y.size == 0:
        raise HTTPException(status_code=400, detail="Audio vacío")

    duracion = len(y) / sr
    if duracion > MAX_AUDIO_SECONDS:
        raise HTTPException(status_code=400, detail="Audio demasiado largo")

    max_abs = np.max(np.abs(y))
    if max_abs > 0:
        y = y / max_abs

    rms = librosa.feature.rms(y=y)[0]
    thr = max(0.02, float(np.mean(rms) * 0.6))
    voiced = rms > thr

    speech_ratio = float(np.mean(voiced))

    pausas = np.sum(~voiced)
    pausas_ratio = pausas / len(voiced)

    score = clamp(
        0.6 * speech_ratio +
        0.4 * (1 - pausas_ratio)
    )

    return {
        "duracion": round(duracion, 2),
        "speech_ratio": round(speech_ratio, 3),
        "pausas_ratio": round(pausas_ratio, 3),
        "score": round(score, 3)
    }


def clasificar(score):
    if score >= 0.7:
        return "ALTO"
    if score >= 0.45:
        return "MEDIO"
    return "BAJO"


# =========================
# UI
# =========================

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h1>Fluidez Lite</h1>
    <form action="/evaluar" method="post" enctype="multipart/form-data">
        Nombre:<br>
        <input name="student"><br><br>
        Audio (.wav):<br>
        <input type="file" name="audio"><br><br>
        <button>Evaluar</button>
    </form>
    """


@app.post("/evaluar", response_class=HTMLResponse)
async def evaluar(student: str = Form(...), audio: UploadFile = File(...)):

    validar_extension(audio.filename)

    uid = uuid.uuid4().hex
    path = UPLOAD_DIR / f"{uid}.wav"

    try:
        await guardar_upload(audio, path)
        m = analizar_audio(path)
        nivel = clasificar(m["score"])

    finally:
        if path.exists():
            os.remove(path)

    return f"""
    <h2>Resultado</h2>
    <p><b>Estudiante:</b> {html.escape(student)}</p>
    <p><b>Score:</b> {m['score']}</p>
    <p><b>Nivel:</b> {nivel}</p>
    <p><b>Duración:</b> {m['duracion']} s</p>
    <p><b>Speech ratio:</b> {m['speech_ratio']}</p>
    <p><b>Pausas ratio:</b> {m['pausas_ratio']}</p>
    <br><a href="/">Volver</a>
    """
