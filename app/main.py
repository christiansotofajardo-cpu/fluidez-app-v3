from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from pathlib import Path
import shutil
import html

import librosa
import numpy as np

app = FastAPI(title="Fluidez App Pro")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg"}


def validar_extension(filename: str):
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Extensión no permitida: {ext}. Usa {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )


def clasificar(valor: float | None):
    if valor is None:
        return None
    if valor >= 0.75:
        return "ALTO"
    if valor >= 0.40:
        return "MEDIO"
    return "BAJO"


def perfil_integrado(nf: str | None, nd: str | None):
    if nf and nd:
        return f"Fluidez {nf} - Decodificación {nd}"
    if nf:
        return f"Solo Fluidez {nf}"
    if nd:
        return f"Solo Decodificación {nd}"
    return "Sin datos"


def tipo_problema(nf: str | None, nd: str | None):
    if nf == "BAJO" and nd == "BAJO":
        return "Dificultad mixta"
    if nf == "BAJO":
        return "Dificultad predominante en fluidez"
    if nd == "BAJO":
        return "Dificultad predominante en decodificación"
    return "Sin dificultades relevantes"


def estado_lectura(nf: str | None, nd: str | None):
    if nf == "ALTO" and nd == "ALTO":
        return "Consolidado"
    if nf == "BAJO" and nd == "BAJO":
        return "Descendido"
    if nf == "MEDIO" or nd == "MEDIO":
        return "En desarrollo"
    if nf == "BAJO" or nd == "BAJO":
        return "En riesgo"
    return "Indeterminado"


def interpretar_resultado(nf: str | None, nd: str | None):
    if nf == "BAJO" and nd == "BAJO":
        return (
            "Se observan dificultades tanto en la precisión para reconocer palabras como en la fluidez de la lectura. "
            "Esto sugiere un proceso lector aún poco automatizado."
        )
    if nf == "BAJO" and nd in {"MEDIO", "ALTO"}:
        return (
            "La precisión lectora aparece relativamente preservada, pero la lectura todavía se realiza con lentitud "
            "o con poca continuidad."
        )
    if nd == "BAJO" and nf in {"MEDIO", "ALTO"}:
        return (
            "La lectura presenta dificultades en el reconocimiento preciso de palabras, lo que puede afectar el rendimiento "
            "lector general."
        )
    if nf == "MEDIO" or nd == "MEDIO":
        return (
            "El desempeño lector se encuentra en una zona intermedia. Hay avances, pero todavía conviene reforzar "
            "algunos componentes para consolidar la lectura."
        )
    if nf == "ALTO" and nd == "ALTO":
        return "El desempeño lector aparece bien consolidado tanto en fluidez como en decodificación."
    return "No fue posible generar una interpretación precisa."


def sugerencia_pedagogica(nf: str | None, nd: str | None):
    if nf == "BAJO" and nd == "BAJO":
        return (
            "Se recomienda práctica guiada frecuente, con apoyo en reconocimiento de palabras y lectura oral breve "
            "para fortalecer precisión y automatización."
        )
    if nf == "BAJO":
        return (
            "Se recomienda reforzar lectura en voz alta con acompañamiento, buscando mejorar ritmo, continuidad y seguridad."
        )
    if nd == "BAJO":
        return (
            "Se recomienda trabajar reconocimiento preciso de palabras, lectura de sílabas y correspondencias grafema-sonido."
        )
    if nf == "MEDIO" or nd == "MEDIO":
        return (
            "Se recomienda continuar con práctica lectora regular y seguimiento para consolidar el desempeño."
        )
    if nf == "ALTO" and nd == "ALTO":
        return (
            "Se sugiere mantener la práctica lectora y avanzar hacia textos más complejos para seguir desarrollando comprensión y soltura."
        )
    return "Sin sugerencia específica."


def conclusion_principal(nf: str | None, nd: str | None):
    if nf == "ALTO" and nd == "ALTO":
        return "Lectura fluida y precisa para su nivel"
    if nf == "BAJO" and nd == "BAJO":
        return "Se observan dificultades importantes en la lectura"
    if nf == "BAJO":
        return "Se observa una dificultad principal en la fluidez lectora"
    if nd == "BAJO":
        return "Se observa una dificultad principal en la precisión lectora"
    return "La lectura se encuentra en proceso de consolidación"


def descripcion_fluidez(nivel: str | None):
    if nivel == "ALTO":
        return "La lectura oral muestra buen ritmo, continuidad y seguridad."
    if nivel == "MEDIO":
        return "La lectura oral se encuentra en desarrollo y todavía puede ganar continuidad y soltura."
    if nivel == "BAJO":
        return "La lectura oral presenta lentitud, pausas excesivas o poca continuidad."
    return "Sin información suficiente."


def descripcion_decodificacion(nivel: str | None):
    if nivel == "ALTO":
        return "El reconocimiento de palabras aparece preciso y estable."
    if nivel == "MEDIO":
        return "El reconocimiento de palabras está en desarrollo, con necesidad de mayor consolidación."
    if nivel == "BAJO":
        return "Se observan debilidades en el reconocimiento preciso de palabras."
    return "Sin información suficiente."


def fortalezas_alertas(nf: str | None, nd: str | None):
    fortalezas = []
    alertas = []

    if nf == "ALTO":
        fortalezas.append("Buena continuidad y ritmo de lectura.")
    elif nf == "MEDIO":
        alertas.append("La fluidez todavía requiere consolidación.")
    elif nf == "BAJO":
        alertas.append("La fluidez lectora aparece descendida.")

    if nd == "ALTO":
        fortalezas.append("Reconocimiento preciso de palabras.")
    elif nd == "MEDIO":
        alertas.append("La decodificación se encuentra en desarrollo.")
    elif nd == "BAJO":
        alertas.append("La precisión en la lectura de palabras requiere apoyo.")

    if not fortalezas:
        fortalezas.append("Se recomienda seguir observando progresos a medida que avance la práctica lectora.")
    if not alertas:
        alertas.append("No se detectan alertas relevantes en esta evaluación.")

    return fortalezas, alertas


def badge_color(nivel: str | None):
    if nivel == "ALTO":
        return "#d1fae5"
    if nivel == "MEDIO":
        return "#fef3c7"
    if nivel == "BAJO":
        return "#fee2e2"
    return "#e5e7eb"


def estado_color(estado: str):
    estado = estado.lower()
    if estado == "consolidado":
        return "#065f46"
    if estado == "en desarrollo":
        return "#92400e"
    if estado == "en riesgo":
        return "#9a3412"
    if estado == "descendido":
        return "#991b1b"
    return "#374151"


def analizar_audio(path: Path):
    """
    Análisis acústico básico del audio.
    """
    y, sr = librosa.load(str(path), sr=None, mono=True)

    if len(y) == 0:
        return {
            "duracion": 0.0,
            "energia": 0.0,
            "silencio": 1.0,
            "zcr": 0.0,
        }

    duracion = float(librosa.get_duration(y=y, sr=sr))
    energia = float(np.mean(np.abs(y)))
    silencio = float(np.mean(np.abs(y) < 0.01))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))

    return {
        "duracion": duracion,
        "energia": energia,
        "silencio": silencio,
        "zcr": zcr,
    }


def calcular_indice_fluidez(metricas: dict):
    """
    Índice preliminar real de fluidez.
    Penaliza audios muy largos y con mucho silencio.
    """
    duracion = metricas["duracion"]
    silencio = metricas["silencio"]

    # Ajuste simple y controlado
    score = 1.0 - (duracion * 0.05 + silencio * 0.8)
    score = max(0.0, min(1.0, score))
    return round(score, 2)


def calcular_indice_decodificacion(metricas: dict):
    """
    Índice preliminar proxy de decodificación.
    Es todavía exploratorio: usa energía y estabilidad simple.
    """
    energia = metricas["energia"]
    zcr = metricas["zcr"]
    silencio = metricas["silencio"]

    score = (energia * 8.0) + (zcr * 2.0) - (silencio * 0.5)
    score = max(0.0, min(1.0, score))
    return round(score, 2)


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
                max-width: 960px;
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
            h2, h3 {
                margin-top: 0;
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
                Esta versión ya no usa simulación guiada por el usuario. Realiza un análisis acústico básico del audio
                para estimar preliminarmente fluidez y decodificación.
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
                <input type="file" name="fluidez_file" required />

                <label>Archivo de Decodificación</label>
                <input type="file" name="decod_file" required />

                <button type="submit">Evaluar</button>
            </form>

            <p class="small">
                Se valida la extensión del archivo, pero no su nombre.
            </p>
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
    forma: str = Form(...),
    fluidez_file: UploadFile = File(...),
    decod_file: UploadFile = File(...),
):
    student_id_safe = html.escape(student_id.strip())
    forma_safe = html.escape(forma.strip())

    fluidez_name = fluidez_file.filename or "sin_nombre"
    decod_name = decod_file.filename or "sin_nombre"

    validar_extension(fluidez_name)
    validar_extension(decod_name)

    fluidez_path = UPLOAD_DIR / Path(fluidez_name).name
    decod_path = UPLOAD_DIR / Path(decod_name).name

    with open(fluidez_path, "wb") as buffer:
        shutil.copyfileobj(fluidez_file.file, buffer)

    with open(decod_path, "wb") as buffer:
        shutil.copyfileobj(decod_file.file, buffer)

    metricas_f = analizar_audio(fluidez_path)
    metricas_d = analizar_audio(decod_path)

    fluidez_index = calcular_indice_fluidez(metricas_f)
    decod_index = calcular_indice_decodificacion(metricas_d)

    nivel_f = clasificar(fluidez_index)
    nivel_d = clasificar(decod_index)

    perfil = perfil_integrado(nivel_f, nivel_d)
    problema = tipo_problema(nivel_f, nivel_d)
    estado = estado_lectura(nivel_f, nivel_d)
    interpretacion = interpretar_resultado(nivel_f, nivel_d)
    sugerencia = sugerencia_pedagogica(nivel_f, nivel_d)
    conclusion = conclusion_principal(nivel_f, nivel_d)
    desc_fluidez = descripcion_fluidez(nivel_f)
    desc_decod = descripcion_decodificacion(nivel_d)
    fortalezas, alertas = fortalezas_alertas(nivel_f, nivel_d)

    fluidez_path.unlink(missing_ok=True)
    decod_path.unlink(missing_ok=True)

    fluidez_name_safe = html.escape(fluidez_name)
    decod_name_safe = html.escape(decod_name)

    fortalezas_html = "".join([f"<li>{html.escape(x)}</li>" for x in fortalezas])
    alertas_html = "".join([f"<li>{html.escape(x)}</li>" for x in alertas])

    return f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>Resultado | Fluidez</title>
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
            .hero h1 {{
                margin-bottom: 8px;
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
            ul {{
                margin-top: 8px;
                padding-left: 20px;
            }}
            a {{
                display: inline-block;
                margin-top: 22px;
                font-weight: bold;
            }}
            .state {{
                font-weight: bold;
            }}
            .metrics {{
                font-size: 13px;
                color: #475569;
                margin-top: 12px;
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
                <p><strong>Conclusión principal:</strong> {html.escape(conclusion)}</p>
                <p class="meta">
                    <strong>ID o nombre:</strong> {student_id_safe} |
                    <strong>Forma:</strong> {forma_safe}
                </p>
                <p class="meta">
                    <strong>Estado lector:</strong>
                    <span class="state" style="color:{estado_color(estado)};">{html.escape(estado)}</span>
                </p>
            </div>

            <div class="grid">
                <div class="box">
                    <h2>Fluidez</h2>
                    <p><strong>Archivo:</strong> {fluidez_name_safe}</p>
                    <p><strong>Índice:</strong> {fluidez_index}</p>
                    <p>
                        <strong>Nivel:</strong>
                        <span class="badge" style="background:{badge_color(nivel_f)};">{nivel_f}</span>
                    </p>
                    <p><strong>¿Qué significa?</strong> {html.escape(desc_fluidez)}</p>
                    <div class="metrics">
                        <div><strong>Duración:</strong> {round(metricas_f["duracion"], 2)} s</div>
                        <div><strong>Silencio:</strong> {round(metricas_f["silencio"], 3)}</div>
                        <div><strong>Energía:</strong> {round(metricas_f["energia"], 4)}</div>
                    </div>
                </div>

                <div class="box">
                    <h2>Decodificación</h2>
                    <p><strong>Archivo:</strong> {decod_name_safe}</p>
                    <p><strong>Índice:</strong> {decod_index}</p>
                    <p>
                        <strong>Nivel:</strong>
                        <span class="badge" style="background:{badge_color(nivel_d)};">{nivel_d}</span>
                    </p>
                    <p><strong>¿Qué significa?</strong> {html.escape(desc_decod)}</p>
                    <div class="metrics">
                        <div><strong>Duración:</strong> {round(metricas_d["duracion"], 2)} s</div>
                        <div><strong>Silencio:</strong> {round(metricas_d["silencio"], 3)}</div>
                        <div><strong>Energía:</strong> {round(metricas_d["energia"], 4)}</div>
                    </div>
                </div>
            </div>

            <div class="section">
                <h3>Síntesis general</h3>
                <p><strong>Perfil integrado:</strong> {html.escape(perfil)}</p>
                <p><strong>Tipo de situación:</strong> {html.escape(problema)}</p>
                <p><strong>Interpretación:</strong> {html.escape(interpretacion)}</p>
            </div>

            <div class="two-cols">
                <div class="section">
                    <h3>Fortalezas observadas</h3>
                    <ul>
                        {fortalezas_html}
                    </ul>
                </div>

                <div class="section">
                    <h3>Alertas o aspectos a reforzar</h3>
                    <ul>
                        {alertas_html}
                    </ul>
                </div>
            </div>

            <div class="section">
                <h3>Recomendación</h3>
                <p>{html.escape(sugerencia)}</p>
            </div>

            <div class="section">
                <h3>Definiciones simples</h3>
                <p><strong>Fluidez:</strong> capacidad de leer con ritmo, continuidad y cierta naturalidad.</p>
                <p><strong>Decodificación:</strong> capacidad de reconocer correctamente las palabras escritas.</p>
            </div>

            <a href="/">Evaluar otro estudiante</a>
        </div>
    </body>
    </html>
    """
    """
