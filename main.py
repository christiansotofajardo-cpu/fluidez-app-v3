from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from pathlib import Path
import html
import os
import re
import uuid
import unicodedata
from typing import Any

import numpy as np
import soundfile as sf
import librosa

from faster_whisper import WhisperModel
from rapidfuzz import fuzz


app = FastAPI(title="Fluidez - ComunicaLab")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".wav"}
MAX_FILE_SIZE_MB = 20
MAX_AUDIO_SECONDS = 240

# =========================
# TEXTOS DE REFERENCIA
# =========================

TEXTOS_REFERENCIA = {
    "2A": """
El duende vivía en el fondo de un bosque muy antiguo.
Todas las mañanas salía a caminar entre los árboles,
escuchando los sonidos de los pájaros y del viento.
Le gustaba observar cómo la luz del sol cambiaba
a lo largo del día, iluminando distintos rincones del lugar.
Un día encontró una pequeña puerta escondida en un tronco.
Al abrirla, descubrió un pasaje secreto que nunca antes había visto.
Desde ese momento, decidió explorar cada rincón del bosque,
sabiendo que siempre podía encontrar algo nuevo.
""".strip(),
    "2B": """
La hormiga trabajaba sin detenerse durante todo el día.
Desde muy temprano comenzaba a recoger pequeñas hojas
y trozos de comida que encontraba en el camino.
Caminaba largas distancias para llevar todo a su hogar,
donde otras hormigas la esperaban.
Aunque el camino era difícil, nunca se rendía.
Sabía que su esfuerzo era importante para el grupo.
Con el tiempo, lograron reunir suficiente alimento
para enfrentar los días en que no podrían salir.
""".strip(),
}

DECOD_REFERENCIA = {
    "2A": [
        {"item": 1, "texto": "indignado", "tipo": "real"},
        {"item": 2, "texto": "cobujango", "tipo": "pseudo"},
        {"item": 3, "texto": "grabadora", "tipo": "real"},
        {"item": 4, "texto": "indestructible", "tipo": "real"},
        {"item": 5, "texto": "prectun", "tipo": "pseudo"},
        {"item": 6, "texto": "intercambiar", "tipo": "real"},
        {"item": 7, "texto": "asderminotar", "tipo": "pseudo"},
        {"item": 8, "texto": "inventor", "tipo": "real"},
        {"item": 9, "texto": "flesujas", "tipo": "pseudo"},
        {"item": 10, "texto": "pantelirdo", "tipo": "pseudo"},
        {"item": 11, "texto": "independencia", "tipo": "real"},
        {"item": 12, "texto": "gracioso", "tipo": "real"},
        {"item": 13, "texto": "tosperantago", "tipo": "pseudo"},
        {"item": 14, "texto": "lornifaru", "tipo": "pseudo"},
        {"item": 15, "texto": "actuar", "tipo": "real"},
        {"item": 16, "texto": "resajes", "tipo": "pseudo"},
    ],
    "2B": [
        {"item": 1, "texto": "adorable", "tipo": "real"},
        {"item": 2, "texto": "olpictaver", "tipo": "pseudo"},
        {"item": 3, "texto": "publicidad", "tipo": "real"},
        {"item": 4, "texto": "internacional", "tipo": "real"},
        {"item": 5, "texto": "placter", "tipo": "pseudo"},
        {"item": 6, "texto": "diferenciar", "tipo": "real"},
        {"item": 7, "texto": "turotenacer", "tipo": "pseudo"},
        {"item": 8, "texto": "columna", "tipo": "real"},
        {"item": 9, "texto": "rintumos", "tipo": "pseudo"},
        {"item": 10, "texto": "trolastaña", "tipo": "pseudo"},
        {"item": 11, "texto": "intranquilidad", "tipo": "real"},
        {"item": 12, "texto": "valiente", "tipo": "real"},
        {"item": 13, "texto": "custorguljera", "tipo": "pseudo"},
        {"item": 14, "texto": "mucapisa", "tipo": "pseudo"},
        {"item": 15, "texto": "trepar", "tipo": "real"},
        {"item": 16, "texto": "taloncer", "tipo": "pseudo"},
    ],
}

# =========================
# MODELO DE TRANSCRIPCIÓN
# =========================

WHISPER_MODEL_NAME = "small"
_whisper_model: WhisperModel | None = None


def get_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(
            WHISPER_MODEL_NAME,
            device="cpu",
            compute_type="int8",
        )
    return _whisper_model


# =========================
# UTILIDADES
# =========================

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def validar_extension(filename: str):
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Extensión no permitida: {ext}. Usa solo archivos .wav",
        )


async def guardar_upload(upload: UploadFile, destino: Path):
    size = 0
    with open(destino, "wb") as buffer:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_FILE_SIZE_MB * 1024 * 1024:
                raise HTTPException(
                    status_code=400,
                    detail=f"El archivo {upload.filename} supera el máximo de {MAX_FILE_SIZE_MB} MB.",
                )
            buffer.write(chunk)
    await upload.close()


def normalizar_texto(texto: str) -> str:
    texto = texto.lower().strip()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(ch for ch in texto if unicodedata.category(ch) != "Mn")
    texto = re.sub(r"[^a-zñ0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def tokenizar(texto: str) -> list[str]:
    txt = normalizar_texto(texto)
    return txt.split() if txt else []


def contar_repeticiones_adjacentes(tokens: list[str]) -> int:
    if not tokens:
        return 0
    rep = 0
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            rep += 1
    return rep


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
    <title>Error | Fluidez</title>
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
        <h1>No se pudo procesar la evaluación</h1>
        <div class="err">{msg}</div>
        <a href="/">Volver</a>
    </div>
</body>
</html>
"""
    )


# =========================
# ANÁLISIS ACÚSTICO / TEMPORAL
# =========================

def corridas_booleanas(arr_bool: np.ndarray):
    if len(arr_bool) == 0:
        return [], []

    true_runs = []
    false_runs = []

    actual_val = arr_bool[0]
    largo = 1

    for v in arr_bool[1:]:
        if v == actual_val:
            largo += 1
        else:
            if actual_val:
                true_runs.append(largo)
            else:
                false_runs.append(largo)
            actual_val = v
            largo = 1

    if actual_val:
        true_runs.append(largo)
    else:
        false_runs.append(largo)

    return true_runs, false_runs


def analizar_audio_temporal(path_audio: Path) -> dict[str, float]:
    y, sr = sf.read(str(path_audio), always_2d=False)

    if isinstance(y, np.ndarray) and y.ndim > 1:
        y = np.mean(y, axis=1)

    y = np.asarray(y, dtype=np.float32)

    if y.size == 0:
        return {
            "duracion_s": 0.0,
            "speech_ratio": 0.0,
            "pausas_n": 0,
            "pausas_largas_ratio": 1.0,
            "voz_segmentos_n": 0,
            "voz_segmento_medio_s": 0.0,
            "pausa_media_s": 0.0,
            "pausas_por_min": 0.0,
            "rms_media": 0.0,
            "rms_sd": 0.0,
            "zcr_media": 0.0,
            "zcr_sd": 0.0,
            "energia_media_abs": 0.0,
            "tempo_aprox": 0.0,
        }

    duracion_s = len(y) / sr
    if duracion_s > MAX_AUDIO_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=f"El audio dura {round(duracion_s,1)} s. Máximo permitido: {MAX_AUDIO_SECONDS} s.",
        )

    max_abs = np.max(np.abs(y))
    if max_abs > 0:
        y = y / max_abs

    frame_length = 2048
    hop_length = 512
    frame_s = hop_length / sr

    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]

    zcr = librosa.feature.zero_crossing_rate(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]

    rms_media = float(np.mean(rms)) if len(rms) else 0.0
    rms_sd = float(np.std(rms)) if len(rms) else 0.0
    zcr_media = float(np.mean(zcr)) if len(zcr) else 0.0
    zcr_sd = float(np.std(zcr)) if len(zcr) else 0.0
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
        "duracion_s": float(duracion_s),
        "speech_ratio": speech_ratio,
        "pausas_n": int(pausas_n),
        "pausas_largas_ratio": pausas_largas_ratio,
        "voz_segmentos_n": int(voz_segmentos_n),
        "voz_segmento_medio_s": voz_segmento_medio_s,
        "pausa_media_s": pausa_media_s,
        "pausas_por_min": pausas_por_min,
        "rms_media": rms_media,
        "rms_sd": rms_sd,
        "zcr_media": zcr_media,
        "zcr_sd": zcr_sd,
        "energia_media_abs": energia_media_abs,
        "tempo_aprox": tempo_aprox,
    }


def puntaje_temporal_fluidez(metricas: dict[str, float]) -> float:
    return round(
        clamp(
            0.40 * metricas["speech_ratio"]
            + 0.25 * clamp(metricas["voz_segmento_medio_s"] / 1.2)
            + 0.20 * (1.0 - clamp(metricas["pausa_media_s"] / 1.2))
            + 0.15 * (1.0 - metricas["pausas_largas_ratio"])
        ),
        4,
    )


# =========================
# TRANSCRIPCIÓN
# =========================

def transcribir_audio(path_audio: Path) -> str:
    model = get_whisper_model()
    segments, info = model.transcribe(
        str(path_audio),
        language="es",
        beam_size=5,
        vad_filter=True,
    )
    texto = " ".join(seg.text for seg in segments).strip()
    return texto


# =========================
# FLUIDEZ LINGÜÍSTICA
# =========================

def lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    m, n = len(a), len(b)
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def analizar_fluidez_linguistica(forma: str, transcripcion: str) -> dict[str, Any]:
    esperado = TEXTOS_REFERENCIA[forma]
    exp_tokens = tokenizar(esperado)
    tr_tokens = tokenizar(transcripcion)

    if not exp_tokens:
        raise HTTPException(status_code=500, detail=f"No hay texto de referencia para la forma {forma}")

    lcs = lcs_len(exp_tokens, tr_tokens)
    cobertura = lcs / len(exp_tokens) if exp_tokens else 0.0
    precision = lcs / len(tr_tokens) if tr_tokens else 0.0
    omisiones = max(0, len(exp_tokens) - lcs)
    exceso = max(0, len(tr_tokens) - lcs)
    rep_adj = contar_repeticiones_adjacentes(tr_tokens)
    rep_ratio = rep_adj / max(len(tr_tokens), 1)

    score = clamp(
        0.45 * cobertura
        + 0.35 * precision
        + 0.10 * (1.0 - clamp(rep_ratio * 3.0))
        + 0.10 * (1.0 - clamp(exceso / max(len(exp_tokens), 1)))
    )

    return {
        "texto_esperado": esperado,
        "tokens_esperados": len(exp_tokens),
        "tokens_transcritos": len(tr_tokens),
        "lcs": lcs,
        "cobertura": round(cobertura, 4),
        "precision_ajuste": round(precision, 4),
        "omisiones_aprox": omisiones,
        "exceso_aprox": exceso,
        "repeticiones_adjacentes": rep_adj,
        "score_linguistico": round(score, 4),
    }


def clasificar_nivel(score: float) -> str:
    if score >= 0.72:
        return "ALTO"
    if score >= 0.48:
        return "MEDIO"
    return "BAJO"


# =========================
# DECODIFICACIÓN
# =========================

def alinear_decod(expected_items: list[dict[str, Any]], transcripcion: str) -> dict[str, Any]:
    spoken = tokenizar(transcripcion)

    resultados = []
    i = 0

    for item in expected_items:
        esperado = normalizar_texto(item["texto"]).replace(" ", "")

        if i >= len(spoken):
            resultados.append({
                "item": item["item"],
                "esperado": item["texto"],
                "tipo": item["tipo"],
                "producido": "",
                "score": 0,
                "correcto": False,
            })
            continue

        cand1 = spoken[i] if i < len(spoken) else ""
        cand2 = (spoken[i] + spoken[i + 1]) if i + 1 < len(spoken) else ""

        s1 = fuzz.ratio(esperado, cand1)
        s2 = fuzz.ratio(esperado, cand2) if cand2 else -1

        if s2 > s1:
            producido = cand2
            score = s2
            i += 2
        else:
            producido = cand1
            score = s1
            i += 1

        correcto = score >= 85

        resultados.append({
            "item": item["item"],
            "esperado": item["texto"],
            "tipo": item["tipo"],
            "producido": producido,
            "score": int(score),
            "correcto": bool(correcto),
        })

    total = len(resultados)
    correctos = sum(1 for r in resultados if r["correcto"])

    reales = [r for r in resultados if r["tipo"] == "real"]
    pseudos = [r for r in resultados if r["tipo"] == "pseudo"]

    correctos_reales = sum(1 for r in reales if r["correcto"])
    correctos_pseudos = sum(1 for r in pseudos if r["correcto"])

    acc_total = correctos / total if total else 0.0
    acc_reales = correctos_reales / len(reales) if reales else 0.0
    acc_pseudos = correctos_pseudos / len(pseudos) if pseudos else 0.0

    # Damos un poco más de peso a pseudopalabras
    score_decod = clamp(0.40 * acc_reales + 0.60 * acc_pseudos)

    return {
        "resultados": resultados,
        "total_items": total,
        "correctos_total": correctos,
        "acc_total": round(acc_total, 4),
        "acc_reales": round(acc_reales, 4),
        "acc_pseudos": round(acc_pseudos, 4),
        "score_decod": round(score_decod, 4),
    }


# =========================
# INTERPRETACIÓN
# =========================

def conclusion_fluidez(nivel: str) -> str:
    if nivel == "ALTO":
        return "Se observa una lectura continua, con buen ajuste temporal y lingüístico."
    if nivel == "MEDIO":
        return "La lectura muestra un desempeño intermedio, con continuidad parcial y margen de mejora."
    return "Se observan dificultades de continuidad y/o ajuste lingüístico en la lectura."

def conclusion_decod(nivel: str) -> str:
    if nivel == "ALTO":
        return "La decodificación aparece bien lograda en este tamizaje."
    if nivel == "MEDIO":
        return "La decodificación se ubica en una zona intermedia."
    return "La decodificación muestra debilidades relevantes en este tamizaje."

def fortalezas_fluidez(score_temp: float, score_ling: float) -> list[str]:
    out = []
    if score_temp >= 0.72:
        out.append("Buen manejo temporal: pausas y continuidad relativamente adecuadas.")
    if score_ling >= 0.72:
        out.append("Buen ajuste al contenido lingüístico del texto esperado.")
    if not out:
        out.append("Se aprecia margen para consolidar la lectura con mayor apoyo.")
    return out

def alertas_fluidez(score_temp: float, score_ling: float) -> list[str]:
    out = []
    if score_temp < 0.48:
        out.append("El patrón temporal muestra pausas frecuentes, segmentos cortos o baja continuidad.")
    if score_ling < 0.48:
        out.append("La transcripción se aleja del texto esperado, sugiriendo omisiones, desajustes o quiebres.")
    if not out:
        out.append("No se observan alertas importantes en esta lectura.")
    return out

def fortalezas_decod(acc_reales: float, acc_pseudos: float) -> list[str]:
    out = []
    if acc_reales >= 0.72:
        out.append("Buen reconocimiento de palabras reales.")
    if acc_pseudos >= 0.72:
        out.append("Buen desempeño en pseudopalabras.")
    if not out:
        out.append("Hay puntos de apoyo, pero todavía conviene afinar esta dimensión.")
    return out

def alertas_decod(acc_reales: float, acc_pseudos: float) -> list[str]:
    out = []
    if acc_reales < 0.48:
        out.append("Bajo rendimiento en palabras reales.")
    if acc_pseudos < 0.48:
        out.append("Bajo rendimiento en pseudopalabras.")
    if not out:
        out.append("No se observan alertas importantes en esta tarea.")
    return out


# =========================
# HOME
# =========================

@app.get("/", response_class=HTMLResponse)
@app.head("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Fluidez - ComunicaLab</title>
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
        .small {
            color: #4b5563;
            font-size: 14px;
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
    </style>
</head>
<body>
    <div class="card">
        <div class="brand">
            <div class="brand-mark">F</div>
            <div class="brand-text">
                <h1>Fluidez</h1>
                <p>by ComunicaLab</p>
            </div>
        </div>

        <p>Sube un audio de fluidez y uno de decodificación para obtener una evaluación lectora integrada.</p>

        <div class="demo">
            Esta versión usa un análisis híbrido:
            <strong>temporal + lingüístico</strong> para fluidez, y
            <strong>comparación con lista esperada</strong> para decodificación.
            <br><br>
            Formato admitido en esta versión: <strong>.wav</strong>
        </div>

        <form action="/evaluar" method="post" enctype="multipart/form-data">
            <label>ID o nombre del estudiante</label>
            <input type="text" name="student_id" placeholder="Ej: S001 o Nombre del estudiante" required />

            <label>Forma</label>
            <select name="forma" required>
                <option value="2A">2A</option>
                <option value="2B">2B</option>
            </select>

            <label>Archivo de Fluidez</label>
            <input type="file" name="fluidez_file" accept=".wav" required />

            <label>Archivo de Decodificación</label>
            <input type="file" name="decod_file" accept=".wav" required />

            <button type="submit">Evaluar</button>
        </form>

        <p class="small">Máximo por archivo: 20 MB. Duración máxima por audio: 240 segundos.</p>
    </div>
</body>
</html>
"""


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return HTMLResponse(status_code=204, content="")


# =========================
# EVALUACIÓN
# =========================

@app.post("/evaluar", response_class=HTMLResponse)
async def evaluar(
    student_id: str = Form(...),
    forma: str = Form(...),
    fluidez_file: UploadFile = File(...),
    decod_file: UploadFile = File(...),
):
    student_id_safe = html.escape(student_id.strip())
    forma_safe = html.escape(forma.strip())

    if forma not in {"2A", "2B"}:
        return render_error("Forma no válida. Usa 2A o 2B.")

    fluidez_name = fluidez_file.filename or "fluidez.wav"
    decod_name = decod_file.filename or "decod.wav"

    validar_extension(fluidez_name)
    validar_extension(decod_name)

    uid = uuid.uuid4().hex[:8]
    fluidez_path = UPLOAD_DIR / f"{uid}_fluidez.wav"
    decod_path = UPLOAD_DIR / f"{uid}_decod.wav"

    try:
        await guardar_upload(fluidez_file, fluidez_path)
        await guardar_upload(decod_file, decod_path)

        # 1) Fluidez: temporal + lingüística
        met_f = analizar_audio_temporal(fluidez_path)
        txt_f = transcribir_audio(fluidez_path)
        flu_ling = analizar_fluidez_linguistica(forma, txt_f)

        score_temp = puntaje_temporal_fluidez(met_f)
        score_ling = flu_ling["score_linguistico"]
        score_fluidez = round(clamp(0.45 * score_temp + 0.55 * score_ling), 4)
        nivel_fluidez = clasificar_nivel(score_fluidez)

        # 2) Decodificación
        txt_d = transcribir_audio(decod_path)
        decod = alinear_decod(DECOD_REFERENCIA[forma], txt_d)
        score_decod = decod["score_decod"]
        nivel_decod = clasificar_nivel(score_decod)

        fortalezas_f = fortalezas_fluidez(score_temp, score_ling)
        alertas_f = alertas_fluidez(score_temp, score_ling)
        fortalezas_d = fortalezas_decod(decod["acc_reales"], decod["acc_pseudos"])
        alertas_d = alertas_decod(decod["acc_reales"], decod["acc_pseudos"])

    except HTTPException as e:
        return render_error(e.detail)
    except Exception as e:
        return render_error(f"Error interno al analizar los audios: {str(e)}")
    finally:
        if fluidez_path.exists():
            os.remove(fluidez_path)
        if decod_path.exists():
            os.remove(decod_path)

    fortalezas_f_html = "".join(f"<li>{html.escape(x)}</li>" for x in fortalezas_f)
    alertas_f_html = "".join(f"<li>{html.escape(x)}</li>" for x in alertas_f)
    fortalezas_d_html = "".join(f"<li>{html.escape(x)}</li>" for x in fortalezas_d)
    alertas_d_html = "".join(f"<li>{html.escape(x)}</li>" for x in alertas_d)

    filas_decod = []
    for r in decod["resultados"]:
        filas_decod.append(
            f"""
            <tr>
                <td>{r["item"]}</td>
                <td>{html.escape(r["esperado"])}</td>
                <td>{html.escape(r["tipo"])}</td>
                <td>{html.escape(r["producido"])}</td>
                <td>{r["score"]}</td>
                <td>{"Sí" if r["correcto"] else "No"}</td>
            </tr>
            """
        )
    tabla_decod_html = "".join(filas_decod)

    return f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Resultado | Fluidez</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1120px;
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
        .two-cols {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-top: 16px;
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
        .note {{
            font-size: 13px;
            color: #475569;
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            padding: 12px;
            border-radius: 10px;
            margin-top: 14px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
            font-size: 13px;
        }}
        th, td {{
            border: 1px solid #e5e7eb;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background: #f3f4f6;
        }}
        .mono {{
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            white-space: pre-wrap;
            background: #fafafa;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 12px;
        }}
        @media (max-width: 800px) {{
            .grid, .two-cols {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="card">
        <div class="hero">
            <h1>Resultado de evaluación</h1>
            <p><strong>ID o nombre:</strong> {student_id_safe} | <strong>Forma:</strong> {forma_safe}</p>
            <p class="meta"><strong>Fluidez:</strong> <span class="badge" style="background:{badge_color(nivel_fluidez)};">{nivel_fluidez}</span> | 
            <strong>Decodificación:</strong> <span class="badge" style="background:{badge_color(nivel_decod)};">{nivel_decod}</span></p>
        </div>

        <div class="grid">
            <div class="box">
                <h2>Fluidez híbrida</h2>
                <p><strong>Puntaje total:</strong> {score_fluidez}</p>
                <p><strong>Componente temporal:</strong> {round(score_temp, 4)}</p>
                <p><strong>Componente lingüístico:</strong> {round(score_ling, 4)}</p>
                <p><strong>Conclusión:</strong> {html.escape(conclusion_fluidez(nivel_fluidez))}</p>
                <div class="metrics">
                    <div><strong>Duración:</strong> {round(met_f["duracion_s"], 2)} s</div>
                    <div><strong>Speech ratio:</strong> {round(met_f["speech_ratio"], 3)}</div>
                    <div><strong>Voz segmento medio:</strong> {round(met_f["voz_segmento_medio_s"], 3)} s</div>
                    <div><strong>Pausa media:</strong> {round(met_f["pausa_media_s"], 3)} s</div>
                    <div><strong>Pausas/min:</strong> {round(met_f["pausas_por_min"], 2)}</div>
                    <div><strong>Pausas largas ratio:</strong> {round(met_f["pausas_largas_ratio"], 3)}</div>
                    <div><strong>Cobertura textual:</strong> {flu_ling["cobertura"]}</div>
                    <div><strong>Precisión de ajuste:</strong> {flu_ling["precision_ajuste"]}</div>
                    <div><strong>Omisiones aprox:</strong> {flu_ling["omisiones_aprox"]}</div>
                    <div><strong>Repeticiones adyacentes:</strong> {flu_ling["repeticiones_adjacentes"]}</div>
                </div>
            </div>

            <div class="box">
                <h2>Decodificación</h2>
                <p><strong>Puntaje total:</strong> {score_decod}</p>
                <p><strong>Exactitud total:</strong> {decod["acc_total"]}</p>
                <p><strong>Palabras reales:</strong> {decod["acc_reales"]}</p>
                <p><strong>Pseudopalabras:</strong> {decod["acc_pseudos"]}</p>
                <p><strong>Conclusión:</strong> {html.escape(conclusion_decod(nivel_decod))}</p>
                <div class="metrics">
                    <div><strong>Items correctos:</strong> {decod["correctos_total"]} / {decod["total_items"]}</div>
                </div>
            </div>
        </div>

        <div class="two-cols">
            <div class="section">
                <h3>Fortalezas en fluidez</h3>
                <ul>{fortalezas_f_html}</ul>
            </div>
            <div class="section">
                <h3>Alertas en fluidez</h3>
                <ul>{alertas_f_html}</ul>
            </div>
        </div>

        <div class="two-cols">
            <div class="section">
                <h3>Fortalezas en decodificación</h3>
                <ul>{fortalezas_d_html}</ul>
            </div>
            <div class="section">
                <h3>Alertas en decodificación</h3>
                <ul>{alertas_d_html}</ul>
            </div>
        </div>

        <div class="section">
            <h3>Transcripción de fluidez</h3>
            <div class="mono">{html.escape(txt_f)}</div>
        </div>

        <div class="section">
            <h3>Transcripción de decodificación</h3>
            <div class="mono">{html.escape(txt_d)}</div>
        </div>

        <div class="section">
            <h3>Detalle de decodificación</h3>
            <table>
                <thead>
                    <tr>
                        <th>Item</th>
                        <th>Esperado</th>
                        <th>Tipo</th>
                        <th>Producido</th>
                        <th>Score</th>
                        <th>Correcto</th>
                    </tr>
                </thead>
                <tbody>
                    {tabla_decod_html}
                </tbody>
            </table>
        </div>

        <div class="section">
            <h3>Nota metodológica</h3>
            <p>
                La fluidez se calcula como un índice híbrido: 
                <strong>45% temporal</strong> y <strong>55% lingüístico</strong>.
                La decodificación se calcula comparando la transcripción con la lista esperada de palabras y pseudopalabras de la forma seleccionada.
            </p>
        </div>

        <a href="/">Evaluar otro estudiante</a>
    </div>
</body>
</html>
"""

