
"""LÉGORA Local — bilingual experimental research demo (2026-11-17).

IMPORTANT: English is an unvalidated extension. Scores and levels are heuristic,
not standardized, diagnostic, or suitable for high-stakes educational decisions.
Only use test recordings with the speaker's explicit authorization.
"""

from __future__ import annotations

import html
import os
import re
import unicodedata
import uuid
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from faster_whisper import WhisperModel
from rapidfuzz import fuzz

app = FastAPI(title="LÉGORA Local — ES/EN (experimental)")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {".wav"}
MAX_FILE_SIZE_MB = 20
MAX_AUDIO_SECONDS = 240
WHISPER_MODEL_NAME = "tiny"  # multilingual model; CPU int8
_whisper_model: WhisperModel | None = None

# Demo-only stimuli. English stimuli are newly proposed, NOT validated/equated
# against Spanish stimuli and NOT suitable for norm-referenced comparisons.
TEXTOS_REFERENCIA: dict[str, dict[str, str]] = {
    "es": {
        "2A": """El duende vivía en el fondo de un bosque muy antiguo.
Todas las mañanas salía a caminar entre los árboles,
escuchando los sonidos de los pájaros y del viento.
Le gustaba observar cómo la luz del sol cambiaba
a lo largo del día, iluminando distintos rincones del lugar.
Un día encontró una pequeña puerta escondida en un tronco.
Al abrirla, descubrió un pasaje secreto que nunca antes había visto.
Desde ese momento, decidió explorar cada rincón del bosque,
sabiendo que siempre podía encontrar algo nuevo.""",
        "2B": """La hormiga trabajaba sin detenerse durante todo el día.
Desde muy temprano comenzaba a recoger pequeñas hojas
y trozos de comida que encontraba en el camino.
Caminaba largas distancias para llevar todo a su hogar,
donde otras hormigas la esperaban.
Aunque el camino era difícil, nunca se rendía.
Sabía que su esfuerzo era importante para el grupo.
Con el tiempo, lograron reunir suficiente alimento
para enfrentar los días en que no podrían salir.""",
    },
    "en": {
        "2A": """A small fox lived near a quiet forest. Every morning, it walked
beside the stream and listened to the birds. One day, the fox found
a little wooden box under an old tree. The box was empty, but there
was a note inside. The note said to look for something beautiful.
The fox walked through the forest and saw flowers, rocks, and bright
sunlight on the water. At sunset, it understood that the forest itself
was full of beautiful things. The fox returned home and told its
friends about the discovery.""",
        "2B": """Maya and her grandfather planted vegetables in a small garden.
They prepared the soil, placed seeds in the ground, and watered them
carefully. Every afternoon, Maya checked for new leaves. Some days
nothing seemed to change, but her grandfather reminded her that
plants needed time to grow. After several weeks, green shoots appeared.
Maya smiled when she saw the first tomato. She learned that patient
work could bring good results. At the end of summer, they shared
the vegetables with their neighbors.""",
    },
}

DECOD_REFERENCIA: dict[str, dict[str, list[dict[str, Any]]]] = {
    "es": {
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
    },
    "en": {
        "2A": [
            {"item": 1, "texto": "garden", "tipo": "real"},
            {"item": 2, "texto": "nembick", "tipo": "pseudo"},
            {"item": 3, "texto": "window", "tipo": "real"},
            {"item": 4, "texto": "remember", "tipo": "real"},
            {"item": 5, "texto": "plomper", "tipo": "pseudo"},
            {"item": 6, "texto": "sunshine", "tipo": "real"},
            {"item": 7, "texto": "vostril", "tipo": "pseudo"},
            {"item": 8, "texto": "blanket", "tipo": "real"},
            {"item": 9, "texto": "frindle", "tipo": "pseudo"},
            {"item": 10, "texto": "taspin", "tipo": "pseudo"},
            {"item": 11, "texto": "children", "tipo": "real"},
            {"item": 12, "texto": "wonderful", "tipo": "real"},
            {"item": 13, "texto": "drimlet", "tipo": "pseudo"},
            {"item": 14, "texto": "climpo", "tipo": "pseudo"},
            {"item": 15, "texto": "discover", "tipo": "real"},
            {"item": 16, "texto": "meplick", "tipo": "pseudo"},
        ],
        "2B": [
            {"item": 1, "texto": "picture", "tipo": "real"},
            {"item": 2, "texto": "wemplen", "tipo": "pseudo"},
            {"item": 3, "texto": "morning", "tipo": "real"},
            {"item": 4, "texto": "important", "tipo": "real"},
            {"item": 5, "texto": "glimter", "tipo": "pseudo"},
            {"item": 6, "texto": "carefully", "tipo": "real"},
            {"item": 7, "texto": "prondick", "tipo": "pseudo"},
            {"item": 8, "texto": "yellow", "tipo": "real"},
            {"item": 9, "texto": "soblent", "tipo": "pseudo"},
            {"item": 10, "texto": "tusper", "tipo": "pseudo"},
            {"item": 11, "texto": "adventure", "tipo": "real"},
            {"item": 12, "texto": "helpful", "tipo": "real"},
            {"item": 13, "texto": "flotven", "tipo": "pseudo"},
            {"item": 14, "texto": "minderp", "tipo": "pseudo"},
            {"item": 15, "texto": "reading", "tipo": "real"},
            {"item": 16, "texto": "drembal", "tipo": "pseudo"},
        ],
    },
}

TEXT = {
    "es": {
        "title": "LÉGORA Local", "subtitle": "Demostración experimental · Español / English",
        "intro": "Evaluación exploratoria de fluidez lectora y decodificación mediante IA pequeña.",
        "language": "Idioma de la evaluación", "form": "Forma", "student": "Código de prueba (no ingresar nombres reales)",
        "fluency": "Audio de fluidez (.wav)", "decoding": "Audio de decodificación (.wav)",
        "submit": "Evaluar lectura", "reference": "Texto de lectura para esta demostración", "wordlist": "Lista de palabras y pseudopalabras",
        "limit": "Máximo 20 MB y 240 segundos por archivo. Uso con audio adulto autorizado o sintético.",
        "warning": "Beta experimental SIN validación bilingüe. No usar para diagnóstico, clasificación individual ni decisiones educativas de alto impacto. Revisar las transcripciones manualmente.",
        "result": "Resultados exploratorios", "return": "Evaluar otra lectura", "id": "Código",
        "overall": "Síntesis", "fluency_result": "Fluidez lectora", "decoding_result": "Decodificación",
        "level": "Nivel heurístico", "score": "Puntaje exploratorio", "dimension": "Dimensiones",
        "rhythm": "Ritmo de lectura", "continuity": "Fluidez continua", "fidelity": "Fidelidad al texto", "accuracy": "Exactitud lectora",
        "real_words": "Lectura de palabras", "nonwords": "Lectura de pseudopalabras",
        "temporal": "Base temporal", "linguistic": "Base lingüística",
        "items": "Ítems correctos (estimación)", "speech": "Proporción de segmentos con energía", "pauses": "Pausas aproximadas por minuto",
        "transcript_f": "Transcripción automática: fluidez", "transcript_d": "Transcripción automática: decodificación",
        "details": "Detalle exploratorio de decodificación", "expected": "Esperado", "produced": "Reconocido", "type": "Tipo",
        "match": "Coincidencia", "correct": "¿Coincide?", "yes": "Sí", "no": "No",
        "notice": "Los umbrales de esta demo son heurísticos y no cuentan con baremos ni validación independiente. Un error del reconocedor puede confundirse con un error de lectura; verificar manualmente.",
        "file_err": "Solo se admiten archivos .wav", "limit_err": "El archivo supera el límite permitido",
        "duration_err": "El audio supera 240 segundos", "form_err": "Forma o idioma no válidos",
        "internal_err": "No se pudo procesar el audio de prueba.",
    },
    "en": {
        "title": "LÉGORA Local", "subtitle": "Experimental demonstration · Español / English",
        "intro": "Exploratory oral-reading fluency and decoding assessment using small AI models.",
        "language": "Assessment language", "form": "Form", "student": "Test code (do not enter real names)",
        "fluency": "Fluency audio (.wav)", "decoding": "Decoding audio (.wav)",
        "submit": "Analyze reading", "reference": "Reading passage for this demonstration", "wordlist": "Word and pseudoword list",
        "limit": "Maximum 20 MB and 240 seconds per file. Use authorized adult or synthetic audio only.",
        "warning": "UNVALIDATED experimental beta. Not for diagnosis, individual classification or high-stakes educational decisions. Manually review the transcripts.",
        "result": "Exploratory results", "return": "Analyze another reading", "id": "Code",
        "overall": "Overview", "fluency_result": "Reading fluency", "decoding_result": "Decoding",
        "level": "Heuristic level", "score": "Exploratory score", "dimension": "Dimensions",
        "rhythm": "Reading rhythm", "continuity": "Reading continuity", "fidelity": "Text correspondence", "accuracy": "Reading accuracy proxy",
        "real_words": "Real-word reading", "nonwords": "Pseudoword reading",
        "temporal": "Temporal component", "linguistic": "Linguistic component",
        "items": "Correct items (estimate)", "speech": "Proportion of high-energy frames", "pauses": "Approximate pauses per minute",
        "transcript_f": "Automatic transcript: fluency", "transcript_d": "Automatic transcript: decoding",
        "details": "Exploratory decoding breakdown", "expected": "Expected", "produced": "Recognized", "type": "Type",
        "match": "Similarity", "correct": "Match?", "yes": "Yes", "no": "No",
        "notice": "This demo uses heuristic thresholds without norms or independent validation. Recognition errors may be mistaken for reading errors; manual review is required.",
        "file_err": "Only .wav files are accepted", "limit_err": "File exceeds the size limit",
        "duration_err": "Audio exceeds 240 seconds", "form_err": "Invalid form or language",
        "internal_err": "The test audio could not be processed.",
    },
}

LEVEL = {
    "es": {"ALTO": "ALTO", "MEDIO": "MEDIO", "BAJO": "BAJO"},
    "en": {"ALTO": "HIGH", "MEDIO": "MEDIUM", "BAJO": "LOW"},
}


def tr(lang: str) -> dict[str, str]:
    return TEXT.get(lang, TEXT["es"])


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def normalizar_texto(texto: str) -> str:
    texto = unicodedata.normalize("NFD", texto.lower().strip())
    texto = "".join(ch for ch in texto if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", texto)).strip()


def tokenizar(texto: str) -> list[str]:
    normalized = normalizar_texto(texto)
    return normalized.split() if normalized else []


def corridas_booleanas(arr_bool: np.ndarray) -> tuple[list[int], list[int]]:
    if len(arr_bool) == 0:
        return [], []
    true_runs: list[int] = []
    false_runs: list[int] = []
    actual = bool(arr_bool[0])
    count = 1
    for value in arr_bool[1:]:
        if bool(value) == actual:
            count += 1
        else:
            (true_runs if actual else false_runs).append(count)
            actual = bool(value)
            count = 1
    (true_runs if actual else false_runs).append(count)
    return true_runs, false_runs


def get_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(WHISPER_MODEL_NAME, device="cpu", compute_type="int8")
    return _whisper_model


def transcribir_audio(path_audio: Path, idioma: str) -> str:
    if idioma not in TEXTOS_REFERENCIA:
        raise ValueError("Unsupported language")
    segments, _ = get_whisper_model().transcribe(
        str(path_audio), language=idioma, beam_size=5, vad_filter=True
    )
    return " ".join(segment.text for segment in segments).strip()


def analizar_audio_temporal(path_audio: Path) -> dict[str, float]:
    y, sr = sf.read(str(path_audio), always_2d=False)
    if isinstance(y, np.ndarray) and y.ndim > 1:
        y = np.mean(y, axis=1)
    y = np.asarray(y, dtype=np.float32)
    if sr <= 0 or not y.size:
        raise ValueError("Invalid or empty WAV")
    duracion_s = len(y) / sr
    if duracion_s > MAX_AUDIO_SECONDS:
        raise HTTPException(status_code=400, detail="Audio exceeds 240 seconds")
    max_abs = float(np.max(np.abs(y)))
    if max_abs > 0:
        y = y / max_abs
    frame_length = 2048
    hop_length = 512
    frame_s = hop_length / sr
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    thr = max(0.02, float(np.mean(rms) * 0.6)) if len(rms) else 0.02
    high_energy = rms > thr if len(rms) else np.array([], dtype=bool)
    speech_ratio = float(np.mean(high_energy)) if len(high_energy) else 0.0
    voz_runs, pausa_runs = corridas_booleanas(high_energy)
    voz_segmento_medio_s = float(np.mean(voz_runs) * frame_s) if voz_runs else 0.0
    pausa_media_s = float(np.mean(pausa_runs) * frame_s) if pausa_runs else 0.0
    min_pausa_larga_frames = max(1, int(0.35 / frame_s))
    pausas_largas_ratio = float(sum(p >= min_pausa_larga_frames for p in pausa_runs) / len(pausa_runs)) if pausa_runs else 0.0
    return {
        "duracion_s": float(duracion_s),
        "speech_ratio": speech_ratio,  # high-energy proxy; NOT validated VAD
        "pausas_n": float(len(pausa_runs)),
        "pausas_largas_ratio": pausas_largas_ratio,
        "voz_segmentos_n": float(len(voz_runs)),
        "voz_segmento_medio_s": voz_segmento_medio_s,
        "pausa_media_s": pausa_media_s,
        "pausas_por_min": float(len(pausa_runs) / (duracion_s / 60.0)),
        "rms_media": float(np.mean(rms)) if len(rms) else 0.0,
        "rms_sd": float(np.std(rms)) if len(rms) else 0.0,
        "zcr_media": float(np.mean(zcr)) if len(zcr) else 0.0,
        "zcr_sd": float(np.std(zcr)) if len(zcr) else 0.0,
    }


def puntaje_temporal_fluidez(m: dict[str, float]) -> float:
    return round(clamp(
        0.40 * m["speech_ratio"]
        + 0.25 * clamp(m["voz_segmento_medio_s"] / 1.2)
        + 0.20 * (1.0 - clamp(m["pausa_media_s"] / 1.2))
        + 0.15 * (1.0 - m["pausas_largas_ratio"])
    ), 4)


def lcs_len(a: list[str], b: list[str]) -> int:
    dp = [0] * (len(b) + 1)
    for item in a:
        prev = 0
        for j, bj in enumerate(b, 1):
            temp = dp[j]
            if item == bj:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[len(b)]


def contar_repeticiones_adjacentes(tokens: list[str]) -> int:
    return sum(tokens[i] == tokens[i - 1] for i in range(1, len(tokens)))


def analizar_fluidez_linguistica(forma: str, transcripcion: str, idioma: str) -> dict[str, Any]:
    esperado = TEXTOS_REFERENCIA[idioma][forma]
    exp_tokens = tokenizar(esperado)
    tr_tokens = tokenizar(transcripcion)
    lcs = lcs_len(exp_tokens, tr_tokens)
    cobertura = lcs / len(exp_tokens) if exp_tokens else 0.0
    precision = lcs / len(tr_tokens) if tr_tokens else 0.0
    omisiones = max(0, len(exp_tokens) - lcs)
    exceso = max(0, len(tr_tokens) - lcs)
    rep_adj = contar_repeticiones_adjacentes(tr_tokens)
    rep_ratio = rep_adj / max(1, len(tr_tokens))
    score = clamp(
        0.45 * cobertura + 0.35 * precision
        + 0.10 * (1.0 - clamp(rep_ratio * 3.0))
        + 0.10 * (1.0 - clamp(exceso / max(1, len(exp_tokens))))
    )
    return {
        "texto_esperado": esperado,
        "tokens_esperados": len(exp_tokens), "tokens_transcritos": len(tr_tokens),
        "lcs": lcs, "cobertura": round(cobertura, 4),
        "precision_ajuste": round(precision, 4),
        "omisiones_aprox": omisiones, "exceso_aprox": exceso,
        "repeticiones_adjacentes": rep_adj,
        "score_linguistico": round(score, 4),
    }


def alinear_decod(expected_items: list[dict[str, Any]], transcripcion: str) -> dict[str, Any]:
    spoken = tokenizar(transcripcion)
    resultados = []
    i = 0
    for item in expected_items:
        esperado = normalizar_texto(item["texto"]).replace(" ", "")
        if i >= len(spoken):
            resultados.append({**item, "producido": "", "score": 0, "correcto": False})
            continue
        cand1 = spoken[i]
        cand2 = spoken[i] + spoken[i + 1] if i + 1 < len(spoken) else ""
        s1 = fuzz.ratio(esperado, cand1)
        s2 = fuzz.ratio(esperado, cand2) if cand2 else -1
        if s2 > s1:
            producido, score = cand2, s2
            i += 2
        else:
            producido, score = cand1, s1
            i += 1
        resultados.append({**item, "producido": producido, "score": int(score), "correcto": bool(score >= 85)})
    reales = [r for r in resultados if r["tipo"] == "real"]
    pseudos = [r for r in resultados if r["tipo"] == "pseudo"]
    acc_reales = sum(r["correcto"] for r in reales) / len(reales) if reales else 0.0
    acc_pseudos = sum(r["correcto"] for r in pseudos) / len(pseudos) if pseudos else 0.0
    return {
        "resultados": resultados, "total_items": len(resultados),
        "correctos_total": sum(r["correcto"] for r in resultados),
        "acc_reales": round(acc_reales, 4), "acc_pseudos": round(acc_pseudos, 4),
        "score_decod": round(clamp(0.40 * acc_reales + 0.60 * acc_pseudos), 4),
    }


def clasificar_nivel(score: float) -> str:
    # Retained from Spanish beta only for visual continuity; NOT norm-referenced.
    return "ALTO" if score >= 0.72 else "MEDIO" if score >= 0.48 else "BAJO"


# Qualitative guidance retained as non-normative, exploratory statements.
INTERP = {
    "es": {
        "f_high": "La lectura muestra continuidad y correspondencia con el texto en esta tarea de prueba.",
        "f_mid": "La lectura presenta continuidad parcial y algunos desajustes que requieren revisión.",
        "f_low": "La lectura presenta interrupciones o diferencias respecto del texto; verificar el audio y la transcripción.",
        "d_high": "Alta correspondencia estimada en esta tarea de palabras y pseudopalabras.",
        "d_mid": "Correspondencia estimada intermedia; revisar la transcripción automática.",
        "d_low": "Baja correspondencia estimada; puede reflejar errores de reconocimiento y requiere revisión humana.",
        "strengths": "Aspectos favorables observados", "alerts": "Aspectos que requieren revisión",
        "continuity_good": "Mayor continuidad temporal en la grabación.",
        "text_good": "Mayor correspondencia automática con el texto de referencia.",
        "words_good": "Mayor correspondencia automática en palabras reales.",
        "nonwords_good": "Mayor correspondencia automática en pseudopalabras.",
        "continuity_warn": "Pausas o fragmentaciones frecuentes según estimación energética.",
        "text_warn": "Desajustes entre transcripción y texto; verificar errores del reconocedor.",
        "words_warn": "Revisar la lectura y el reconocimiento de palabras reales.",
        "nonwords_warn": "Revisar especialmente pseudopalabras: el ASR puede sustituirlas por palabras reales.",
        "no_f": "Sin fortalezas identificables mediante estos umbrales experimentales.",
        "no_a": "Sin alertas automáticas bajo estos umbrales experimentales.",
    },
    "en": {
        "f_high": "The recording shows relatively continuous reading and correspondence with the reference passage in this test.",
        "f_mid": "The recording shows partial continuity and some differences requiring review.",
        "f_low": "The recording shows interruptions or text differences; check both audio and automatic transcript.",
        "d_high": "Higher estimated match on this real-word and pseudoword task.",
        "d_mid": "Intermediate estimated match; review automatic recognition output.",
        "d_low": "Lower estimated match, potentially affected by recognition errors; human review is essential.",
        "strengths": "Observed strengths (exploratory)", "alerts": "Aspects requiring review",
        "continuity_good": "Greater temporal continuity in the recording.",
        "text_good": "Greater automatic correspondence with the reference passage.",
        "words_good": "Greater automatic correspondence for real words.",
        "nonwords_good": "Greater automatic correspondence for pseudowords.",
        "continuity_warn": "Frequent pauses or breaks under the energy-based estimate.",
        "text_warn": "Differences between transcript and passage; check speech-recognition errors.",
        "words_warn": "Review real-word reading and recognition.",
        "nonwords_warn": "Review pseudowords especially: speech recognizers may replace them with real words.",
        "no_f": "No strengths identified by these experimental thresholds.",
        "no_a": "No automated alerts under these experimental thresholds.",
    },
}


def orientaciones(score_temp: float, score_ling: float, decod: dict[str, Any], idioma: str):
    r = INTERP[idioma]
    strengths = []
    alerts = []
    if score_temp >= 0.72:
        strengths.append(r["continuity_good"])
    if score_ling >= 0.72:
        strengths.append(r["text_good"])
    if decod["acc_reales"] >= 0.72:
        strengths.append(r["words_good"])
    if decod["acc_pseudos"] >= 0.72:
        strengths.append(r["nonwords_good"])
    if score_temp < 0.48:
        alerts.append(r["continuity_warn"])
    if score_ling < 0.48:
        alerts.append(r["text_warn"])
    if decod["acc_reales"] < 0.48:
        alerts.append(r["words_warn"])
    if decod["acc_pseudos"] < 0.48:
        alerts.append(r["nonwords_warn"])
    return strengths or [r["no_f"]], alerts or [r["no_a"]]


def layout(body: str, idioma: str) -> HTMLResponse:
    t = tr(idioma)
    page = f"""<!doctype html><html lang="{html.escape(idioma)}"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{t['title']} · Demo ES/EN</title>
<style>
body {{font-family:system-ui,-apple-system,sans-serif;background:#f8f8f8;color:#171717;margin:0;line-height:1.55}}
main {{max-width:1000px;margin:32px auto;padding:24px}}
.card {{border:1px solid #d1d1d1;border-radius:12px;padding:24px;background:white;margin:18px 0}}
h1,h2,h3 {{margin-top:0}}label {{display:block;margin:10px 0 4px;font-weight:600}}
input,select,button {{font:inherit;border:1px solid #bbb;border-radius:8px;padding:10px;max-width:100%;box-sizing:border-box}}
input[type=file],input[type=text],select {{width:100%}}
button {{background:#222;color:white;border-color:#222;cursor:pointer;margin-top:18px}}
.small {{color:#555;font-size:.92em}}.warning {{border-left:4px solid #333;background:#f5f5f5;padding:12px;margin:15px 0}}
.grid {{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}
pre {{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f6f6;padding:12px;border:1px solid #ddd;border-radius:8px}}
table {{width:100%;border-collapse:collapse;font-size:.9em}}th,td {{padding:7px;border-bottom:1px solid #ddd;text-align:left}}
a {{color:#111;text-underline-offset:3px}}.lang-nav {{float:right}}
@media(max-width:700px) {{.grid {{grid-template-columns:1fr}} main {{padding:12px}}}}
</style></head><body><main>
<div class="lang-nav"><a href="/?lang=es">Español</a> | <a href="/?lang=en">English</a></div>
<h1>{t['title']}</h1><p class="small">{t['subtitle']}</p><p>{t['intro']}</p>
<div class="warning"><strong>{t['warning']}</strong></div>
{body}</main></body></html>"""
    return HTMLResponse(content=page, headers={"Cache-Control": "no-store"})


@app.get("/", response_class=HTMLResponse)
def home(lang: str = "es", forma: str = "2A"):
    idioma = lang if lang in TEXTOS_REFERENCIA else "es"
    forma = forma if forma in TEXTOS_REFERENCIA[idioma] else "2A"
    t = tr(idioma)
    passage = html.escape(TEXTOS_REFERENCIA[idioma][forma])
    words = " · ".join(html.escape(item["texto"]) for item in DECOD_REFERENCIA[idioma][forma])
    body = f"""
<div class="card"><form method="get" action="/">
<label for="lang">{t['language']}</label><select id="lang" name="lang">
<option value="es" {'selected' if idioma == 'es' else ''}>Español</option>
<option value="en" {'selected' if idioma == 'en' else ''}>English</option></select>
<label for="forma">{t['form']}</label><select id="forma" name="forma">
<option value="2A" {'selected' if forma == '2A' else ''}>2A</option>
<option value="2B" {'selected' if forma == '2B' else ''}>2B</option></select>
<button type="submit">{'Cambiar' if idioma == 'es' else 'Apply'}</button></form></div>
<div class="card"><h2>{t['reference']}</h2><pre>{passage}</pre>
<h3>{t['wordlist']}</h3><p>{words}</p></div>
<div class="card"><form method="post" action="/evaluar" enctype="multipart/form-data">
<input type="hidden" name="idioma" value="{idioma}">
<input type="hidden" name="forma" value="{forma}">
<label for="student_id">{t['student']}</label>
<input id="student_id" type="text" name="student_id" value="DEMO-001" maxlength="32" required>
<label for="fluidez_file">{t['fluency']}</label><input id="fluidez_file" type="file" name="fluidez_file" accept=".wav" required>
<label for="decod_file">{t['decoding']}</label><input id="decod_file" type="file" name="decod_file" accept=".wav" required>
<button type="submit">{t['submit']}</button><p class="small">{t['limit']}</p>
</form></div>"""
    return layout(body, idioma)


async def guardar_upload(upload: UploadFile, destino: Path, idioma: str) -> None:
    t = tr(idioma)
    if Path(upload.filename or "").suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=t["file_err"])
    size = 0
    with destino.open("wb") as buffer:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_FILE_SIZE_MB * 1024 * 1024:
                raise HTTPException(status_code=400, detail=t["limit_err"])
            buffer.write(chunk)
    await upload.close()


def metric_box(label: str, value: str) -> str:
    return f"<div class='card'><strong>{html.escape(label)}</strong><div>{html.escape(value)}</div></div>"


@app.post("/evaluar", response_class=HTMLResponse)
async def evaluar(
    student_id: str = Form(...),
    idioma: str = Form(...),
    forma: str = Form(...),
    fluidez_file: UploadFile = File(...),
    decod_file: UploadFile = File(...),
):
    idioma = idioma if idioma in TEXTOS_REFERENCIA else "es"
    t = tr(idioma)
    if forma not in TEXTOS_REFERENCIA[idioma]:
        raise HTTPException(status_code=400, detail=t["form_err"])
    # Do not store real names. This is a demo; enforce a short code only.
    code = student_id.strip()[:32]
    if not re.fullmatch(r"[A-Za-z0-9_-]{2,32}", code):
        raise HTTPException(status_code=400, detail="Use a short anonymous test code (A-Z, 0-9, underscore or hyphen).")
    uid = uuid.uuid4().hex
    fluidez_path = UPLOAD_DIR / f"{uid}_fluidez.wav"
    decod_path = UPLOAD_DIR / f"{uid}_decod.wav"
    try:
        await guardar_upload(fluidez_file, fluidez_path, idioma)
        await guardar_upload(decod_file, decod_path, idioma)
        met_f = analizar_audio_temporal(fluidez_path)
        txt_f = transcribir_audio(fluidez_path, idioma)
        ling = analizar_fluidez_linguistica(forma, txt_f, idioma)
        score_temp = puntaje_temporal_fluidez(met_f)
        score_ling = ling["score_linguistico"]
        score_fluidez = round(clamp(0.45 * score_temp + 0.55 * score_ling), 4)
        txt_d = transcribir_audio(decod_path, idioma)
        decod = alinear_decod(DECOD_REFERENCIA[idioma][forma], txt_d)
        continuidad = clamp(0.50 * met_f["speech_ratio"]
                            + 0.25 * clamp(met_f["voz_segmento_medio_s"] / 1.2)
                            + 0.25 * (1.0 - clamp(met_f["pausa_media_s"] / 1.2)))
    except HTTPException:
        raise
    except Exception:
        # Do not expose internal error strings or user-submitted data in responses.
        raise HTTPException(status_code=400, detail=t["internal_err"]) from None
    finally:
        for path in (fluidez_path, decod_path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        await fluidez_file.close()
        await decod_file.close()

    dims = [
        (t["rhythm"], score_temp),
        (t["continuity"], continuidad),
        (t["fidelity"], ling["cobertura"]),
        (t["accuracy"], ling["precision_ajuste"]),
        (t["real_words"], decod["acc_reales"]),
        (t["nonwords"], decod["acc_pseudos"]),
    ]
    dimension_html = "".join(
        metric_box(name, f"{round(score, 3)} · {LEVEL[idioma][clasificar_nivel(score)]} ({t['level']})")
        for name, score in dims
    )
    r = INTERP[idioma]
    f_label = ("high" if score_fluidez >= 0.72 else "mid" if score_fluidez >= 0.48 else "low")
    d_label = ("high" if decod["score_decod"] >= 0.72 else "mid" if decod["score_decod"] >= 0.48 else "low")
    strengths, alerts = orientaciones(score_temp, score_ling, decod, idioma)
    strengths_html = "".join(f"<li>{html.escape(item)}</li>" for item in strengths)
    alerts_html = "".join(f"<li>{html.escape(item)}</li>" for item in alerts)
    rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in (
            item["item"], item["texto"], item["tipo"], item["producido"],
            item["score"], t["yes"] if item["correcto"] else t["no"]
        )) + "</tr>" for item in decod["resultados"]
    )
    header = "".join(f"<th>{html.escape(x)}</th>" for x in (
        "#", t["expected"], t["type"], t["produced"], t["match"], t["correct"]
    ))
    body = f"""
<div class="card"><h2>{t['result']}</h2>
<p><strong>{t['id']}:</strong> {html.escape(code)} &nbsp;|&nbsp; {t['language']}: {idioma.upper()} &nbsp;|&nbsp; {t['form']}: {html.escape(forma)}</p>
<div class="grid">
{metric_box(t['fluency_result'], f"{score_fluidez} · {LEVEL[idioma][clasificar_nivel(score_fluidez)]}")}
{metric_box(t['decoding_result'], f"{decod['score_decod']} · {LEVEL[idioma][clasificar_nivel(decod['score_decod'])]}")}
</div><p class="small">{t['temporal']}: {score_temp} · {t['linguistic']}: {score_ling} · {t['items']}: {decod['correctos_total']}/{decod['total_items']}</p>
<p class="small">{t['speech']}: {met_f['speech_ratio']:.3f} · {t['pauses']}: {met_f['pausas_por_min']:.2f}</p>
<p>{html.escape(r['f_' + f_label])} {html.escape(r['d_' + d_label])}</p>
<h3>{t['dimension']}</h3><div class="grid">{dimension_html}</div>
<div class="grid"><div><h3>{html.escape(r['strengths'])}</h3><ul>{strengths_html}</ul></div>
<div><h3>{html.escape(r['alerts'])}</h3><ul>{alerts_html}</ul></div></div>
<h3>{t['transcript_f']}</h3><pre>{html.escape(txt_f)}</pre>
<h3>{t['transcript_d']}</h3><pre>{html.escape(txt_d)}</pre>
<h3>{t['details']}</h3><div style="overflow-x:auto"><table><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table></div>
<div class="warning">{t['notice']}</div>
<p><a href="/?lang={idioma}&forma={html.escape(forma)}">{t['return']}</a></p></div>"""
    return layout(body, idioma)


@app.get("/health")
def health():
    # Health does not prove Whisper has downloaded or that inference works.
    return {"app": "LEGORA Local", "status": "up", "languages": ["es", "en"],
            "version": "0.1-experimental", "model_configured": WHISPER_MODEL_NAME,
            "model_loaded": _whisper_model is not None, "validated": False}
