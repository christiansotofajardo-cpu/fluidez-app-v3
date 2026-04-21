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

TEXTOS_REFERENCIA = {
    "2A": (
        "El duende vivía en el fondo de un bosque muy antiguo. "
        "Todas las mañanas salía a caminar entre los árboles, "
        "escuchando los sonidos de los pájaros y del viento."
    ),
    "2B": (
        "La hormiga trabajaba sin detenerse durante todo el día. "
        "Desde muy temprano comenzaba a recoger pequeñas hojas "
        "y trozos de comida que encontraba en el camino."
    ),
}


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def validar_extension(filename: str):
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Solo se permiten archivos .wav")


async def guardar_upload(upload: UploadFile, destino: Path):
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


def corridas_booleanas(arr_bool: np.ndarray):
    if len(arr_bool) == 0:
        return [], []

    true_runs = []
    false_runs = []

    actual = arr_bool[0]
    largo = 1

    for v in arr_bool[1:]:
        if v == actual:
            largo += 1
        else:
            if actual:
                true_runs.append(largo)
            else:
                false_runs.append(largo)
            actual = v
            largo = 1

    if actual:
        true_runs.append(largo)
    else:
        false_runs.append(largo)

    return true_runs, false_runs


def analizar_audio(path_audio: Path):
    y, sr = sf.read(str(path_audio), always_2d=False)

    if isinstance(y, np.ndarray) and y.ndim > 1:
        y = np.mean(y, axis=1)

    y = np.asarray(y, dtype=np.float32)

    if y.size == 0:
        raise HTTPException(status_code=400, detail="Audio vacío")

    duracion_s = len(y) / sr
    if duracion_s > MAX_AUDIO_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=f"El audio dura {round(duracion_s, 1)} s. Máximo: {MAX_AUDIO_SECONDS} s.",
        )

    max_abs = np.max(np.abs(y))
    if max_abs > 0:
        y = y / max_abs

    frame_length = 2048
    hop_length = 512
    frame_s = hop_length / sr

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(
        y=y, frame_length=frame_length, hop_length=hop_length
    )[0]

    rms_media = float(np.mean(rms)) if len(rms) else 0.0
    rms_sd = float(np.std(rms)) if len(rms) else 0.0
    zcr_media = float(np.mean(zcr)) if len(zcr) else 0.0
    energia_media_abs = float(np.mean(np.abs(y)))

    thr = max(0.02, float(np.mean(rms) * 0.6)) if len(rms) else 0.02
    voiced = rms > thr if len(rms) else np.array([], dtype=bool)

    speech_ratio = float(np.mean(voiced)) if len(voiced) else 0.0

    voz_runs, pausa_runs = corridas_booleanas(voiced)
    voz_segmentos_n = len(voz_runs)
    voz_segmento_medio_s = float(np.mean(voz_runs) * frame_s) if voz_runs else 0.0

    pausas_n = len(pausa_runs)
    pausa_media_s = float(np.mean(pausa_runs) * frame_s) if pausa_runs else 0.0

    min_pausa_larga_s = 0.35
    min_pausa_larga_frames = max(1, int(min_pausa_larga_s / frame_s))
    pausas_largas = [p for p in pausa_runs if p >= min_pausa_larga_frames]
    pausas_largas_ratio = float(len(pausas_largas) / len(pausa_runs)) if pausa_runs else 0.0

    pausas_por_min = float(pausas_n / (duracion_s / 60.0)) if duracion_s > 0 else 0.0

    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        tempo_aprox = float(librosa.feature.tempo(onset_envelope=onset_env, sr=sr)[0])
    except Exception:
        tempo_aprox = 0.0

    return {
        "duracion_s": round(float(duracion_s), 2),
        "speech_ratio": round(speech_ratio, 3),
        "voz_segmentos_n": int(voz_segmentos_n),
        "voz_segmento_medio_s": round(voz_segmento_medio_s, 3),
        "pausas_n": int(pausas_n),
        "pausa_media_s": round(pausa_media_s, 3),
        "pausas_por_min": round(pausas_por_min, 2),
        "pausas_largas_ratio": round(pausas_largas_ratio, 3),
        "rms_media": round(rms_media, 4),
        "rms_sd": round(rms_sd, 4),
        "zcr_media": round(zcr_media, 4),
        "energia_media_abs": round(energia_media_abs, 4),
        "tempo_aprox": round(tempo_aprox, 2),
    }


def puntaje_fluidez(metricas: dict) -> float:
    score = (
        0.35 * metricas["speech_ratio"]
        + 0.25 * clamp(metricas["voz_segmento_medio_s"] / 1.2)
        + 0.20 * (1.0 - clamp(metricas["pausa_media_s"] / 1.0))
        + 0.15 * (1.0 - clamp(metricas["pausas_por_min"] / 25.0))
        + 0.05 * (1.0 - metricas["pausas_largas_ratio"])
    )
    return round(clamp(score), 4)


def clasificar(score: float) -> str:
    if score >= 0.72:
        return "ALTO"
    if score >= 0.48:
        return "MEDIO"
    return "BAJO"


def conclusion_fluidez(nivel: str) -> str:
    if nivel == "ALTO":
        return "Se observa una lectura continua y bien organizada temporalmente."
    if nivel == "MEDIO":
        return "La lectura se ubica en una zona intermedia, con margen de consolidación."
    return "Se observan dificultades temporales de continuidad, ritmo o manejo de pausas."


def fortalezas(metricas: dict, nivel: str) -> list[str]:
    out = []
    if metricas["speech_ratio"] >= 0.60:
        out.append("Buena proporción de habla efectiva.")
    if metricas["voz_segmento_medio_s"] >= 0.60:
        out.append("Segmentos de lectura relativamente continuos.")
    if metricas["pausa_media_s"] <= 0.40:
        out.append("Pausas relativamente breves.")
    if not out and nivel != "BAJO":
        out.append("Se aprecia una base funcional de lectura.")
    if not out:
        out.append("Se recomienda seguir monitoreando con nuevas lecturas.")
    return out


def alertas(metricas: dict, nivel: str) -> list[str]:
    out = []
    if metricas["speech_ratio"] < 0.45:
        out.append("Baja proporción de habla efectiva.")
    if metricas["pausa_media_s"] > 0.60:
        out.append("Pausas medias elevadas.")
    if metricas["pausas_por_min"] > 18:
        out.append("Alta frecuencia de pausas por minuto.")
    if metricas["pausas_largas_ratio"] > 0.45:
        out.append("Alta proporción de pausas largas.")
    if not out and nivel == "ALTO":
        out.append("No se observan alertas relevantes en este tamizaje.")
    if not out:
        out.append("No se observan alertas críticas, aunque conviene seguir monitoreando.")
    return out


def badge_color(nivel: str | None):
    if nivel == "ALTO":
        return "#d1fae5"
    if nivel == "MEDIO":
        return "#fef3c7"
    if nivel == "BAJO":
        return "#fee2e2"
    return "#e5e7eb"


def render_error(message: str):
    msg = html.escape(message)
    return HTMLResponse(
        status_code=400,
        content=f"""
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Error | Fluidez Lite</title>
<style>
body {{
    font-family: Arial, sans-serif;
    max-width: 860px;
    margin: 40px auto;
    padding: 20px;
    background: #f8fafc;
    color: #111827;
}}
.card {{
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 28px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.06);
}}
.err {{
    background: #fef2f2;
    border: 1px solid #fecaca;
    color: #991b1b;
    padding: 16px;
    border-radius: 12px;
}}
a {{
    display: inline-block;
    margin-top: 20px;
    font-weight: bold;
}}
</style>
</head>
<body>
<div class="card">
    <h1>No se pudo procesar el audio</h1>
    <div class="err">{msg}</div>
    <a href="/">Volver</a>
</div>
</body>
</html>
"""
    )


@app.get("/", response_class=HTMLResponse)
@app.head("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Fluidez Lite - ComunicaLab</title>
<style>
body {
    font-family: Arial, sans-serif;
    max-width: 980px;
    margin: 40px auto;
    padding: 20px;
    line-height: 1.5;
    background: #f8fafc;
    color: #111827;
}
.card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 28px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.06);
}
.brand {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 14px;
}
.brand-mark {
    width: 52px;
    height: 52px;
    border-radius: 14px;
    background: linear-gradient(135deg, #1d4ed8, #0f172a);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 26px;
    font-weight: bold;
}
.brand-text h1 {
    margin: 0;
    font-size: 34px;
}
.brand-text p {
    margin: 2px 0 0 0;
    color: #475569;
    font-size: 14px;
}
label {
    display: block;
    margin-top: 8px;
    font-weight: bold;
}
input, button, select {
    padding: 12px;
    margin-top: 6px;
    margin-bottom: 16px;
    width: 100%;
    box-sizing: border-box;
    border-radius: 10px;
    border: 1px solid #d1d5db;
    font-size: 15px;
}
button {
    cursor: pointer;
    font-weight: bold;
    background: linear-gradient(135deg, #0f172a, #1d4ed8);
    color: white;
    border: none;
}
.demo {
    background: #eef2ff;
    color: #3730a3;
    padding: 14px;
    border-radius: 10px;
    font-size: 14px;
    margin-bottom: 18px;
    border: 1px solid #c7d2fe;
}
.small {
    color: #4b5563;
    font-size: 14px;
}
</style>
</head>
<body>
<div class="card">
    <div class="brand">
        <div class="brand-mark">F</div>
        <div class="brand-text">
            <h1>Fluidez Lite</h1>
            <p>by ComunicaLab</p>
        </div>
    </div>

    <p>Versión liviana para hosting gratuito.</p>

    <div class="demo">
        Puedes elegir <strong>modo</strong> y <strong>forma</strong>.
        <br><br>
        <strong>Fluidez</strong>: entrega métricas reales de continuidad temporal.
        <br>
        <strong>Decodificación</strong>: módulo reservado para el motor serio.
    </div>

    <form action="/evaluar" method="post" enctype="multipart/form-data">
        <label>ID o nombre del estudiante</label>
        <input type="text" name="student_id" placeholder="Ej: S001 o Nombre del estudiante" required />

        <label>Modo</label>
        <select name="modo" required>
            <option value="fluidez">Fluidez</option>
            <option value="decodificacion">Decodificación</option>
        </select>

        <label>Forma</label>
        <select name="forma" required>
            <option value="2A">2A</option>
            <option value="2B">2B</option>
        </select>

        <label>Archivo de audio</label>
        <input type="file" name="audio_file" accept=".wav" required />

        <button type="submit">Evaluar</button>
    </form>

    <p class="small">Formato: .wav | máximo: 15 MB | duración máxima: 180 s</p>
</div>
</body>
</html>
"""


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return HTMLResponse(status_code=204, content="")


@app.post("/evaluar", response_class=HTMLResponse)
async def evaluar(
    student_id: str = Form(...),
    modo: str = Form(...),
    forma: str = Form(...),
    audio_file: UploadFile = File(...),
):
    student_id_safe = html.escape(student_id.strip())
    forma_safe = html.escape(forma.strip())
    modo_safe = html.escape(modo.strip())

    if forma not in {"2A", "2B"}:
        return render_error("Forma no válida. Usa 2A o 2B.")

    if modo not in {"fluidez", "decodificacion"}:
        return render_error("Modo no válido.")

    audio_name = audio_file.filename or "audio.wav"
    validar_extension(audio_name)

    uid = uuid.uuid4().hex[:8]
    audio_path = UPLOAD_DIR / f"{uid}.wav"

    try:
        await guardar_upload(audio_file, audio_path)

        if modo == "decodificacion":
            return HTMLResponse(
                content=f"""
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Decodificación | Próximamente</title>
<style>
body {{
    font-family: Arial, sans-serif;
    max-width: 980px;
    margin: 40px auto;
    padding: 20px;
    background: #f8fafc;
    color: #111827;
}}
.card {{
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 28px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.06);
}}
.badge {{
    display: inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    background: #fef3c7;
    font-weight: bold;
}}
a {{
    display: inline-block;
    margin-top: 22px;
    font-weight: bold;
}}
</style>
</head>
<body>
<div class="card">
    <h1>Módulo de decodificación</h1>
    <p><strong>ID o nombre:</strong> {student_id_safe}</p>
    <p><strong>Forma:</strong> {forma_safe}</p>
    <p><strong>Estado:</strong> <span class="badge">Reservado para motor serio</span></p>
    <p>
        Esta versión gratuita está optimizada para fluidez temporal liviana.
        La decodificación completa, especialmente con pseudopalabras y comparación lingüística robusta,
        se ejecutará en la versión seria del sistema.
    </p>
    <a href="/">Volver</a>
</div>
</body>
</html>
"""
            )

        metricas = analizar_audio(audio_path)
        score = puntaje_fluidez(metricas)
        nivel = clasificar(score)
        fortalezas_html = "".join(
            f"<li>{html.escape(x)}</li>" for x in fortalezas(metricas, nivel)
        )
        alertas_html = "".join(
            f"<li>{html.escape(x)}</li>" for x in alertas(metricas, nivel)
        )

    except HTTPException as e:
        return render_error(e.detail)
    except Exception as e:
        return render_error(f"Error interno al analizar el audio: {str(e)}")
    finally:
        if audio_path.exists():
            os.remove(audio_path)

    return f"""
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Resultado | Fluidez Lite</title>
<style>
body {{
    font-family: Arial, sans-serif;
    max-width: 1040px;
    margin: 40px auto;
    padding: 20px;
    line-height: 1.6;
    background: #f8fafc;
    color: #111827;
}}
.card {{
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 28px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.06);
}}
.hero {{
    background: linear-gradient(135deg, #eef2ff, #f8fafc);
    border: 1px solid #dbeafe;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 20px;
}}
.meta {{
    color: #4b5563;
    font-size: 14px;
}}
.grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-top: 20px;
}}
.box {{
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 18px;
    background: #ffffff;
}}
.badge {{
    display: inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    font-weight: bold;
}}
.section {{
    margin-top: 20px;
    background: #f9fafb;
    border-radius: 12px;
    padding: 18px;
    border: 1px solid #e5e7eb;
}}
.metrics {{
    font-size: 13px;
    color: #475569;
    margin-top: 12px;
}}
ul {{
    margin-top: 8px;
    padding-left: 20px;
}}
a {{
    display: inline-block;
    margin-top: 22px;
    font-weight: bold;
}}
@media (max-width: 800px) {{
    .grid {{
        grid-template-columns: 1fr;
    }}
}}
</style>
</head>
<body>
<div class="card">
    <div class="hero">
        <h1>Resultado de evaluación</h1>
        <p><strong>ID o nombre:</strong> {student_id_safe}</p>
        <p class="meta"><strong>Modo:</strong> {modo_safe} | <strong>Forma:</strong> {forma_safe}</p>
        <p><strong>Fluidez:</strong> <span class="badge" style="background:{badge_color(nivel)};">{nivel}</span></p>
        <p><strong>Conclusión principal:</strong> {html.escape(conclusion_fluidez(nivel))}</p>
    </div>

    <div class="grid">
        <div class="box">
            <h2>Índice de fluidez</h2>
            <p><strong>Puntaje:</strong> {score}</p>
            <p><strong>Nivel:</strong> <span class="badge" style="background:{badge_color(nivel)};">{nivel}</span></p>
            <div class="metrics">
                <div><strong>Duración:</strong> {metricas["duracion_s"]} s</div>
                <div><strong>Speech ratio:</strong> {metricas["speech_ratio"]}</div>
                <div><strong>Segmento medio de voz:</strong> {metricas["voz_segmento_medio_s"]} s</div>
                <div><strong>Pausa media:</strong> {metricas["pausa_media_s"]} s</div>
                <div><strong>Pausas por minuto:</strong> {metricas["pausas_por_min"]}</div>
                <div><strong>Pausas largas ratio:</strong> {metricas["pausas_largas_ratio"]}</div>
                <div><strong>Tempo aprox:</strong> {metricas["tempo_aprox"]}</div>
            </div>
        </div>

        <div class="box">
            <h2>Referencia de la forma</h2>
            <p>Esta versión usa la forma seleccionada como marco de tarea, sin transcripción pesada.</p>
            <div class="metrics">
                <div><strong>Forma seleccionada:</strong> {forma_safe}</div>
                <div><strong>Texto de referencia:</strong> {html.escape(TEXTOS_REFERENCIA[forma][:160])}...</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h3>Fortalezas observadas</h3>
        <ul>{fortalezas_html}</ul>
    </div>

    <div class="section">
        <h3>Alertas o aspectos a reforzar</h3>
        <ul>{alertas_html}</ul>
    </div>

    <div class="section">
        <h3>Nota metodológica</h3>
        <p>
            Esta es una versión <strong>liviana</strong> orientada a hosting gratuito.
            Evalúa la fluidez como un <strong>tamizaje temporal</strong>, priorizando continuidad,
            pausas y proporción de habla, sin cargar transcripción pesada.
        </p>
    </div>

    <a href="/">Evaluar otro estudiante</a>
</div>
</body>
</html>
"""
