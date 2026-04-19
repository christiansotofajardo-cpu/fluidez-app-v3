from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from pathlib import Path
import shutil
import random
import html

app = FastAPI(title="Fluidez App v2")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg"}

def detectar_tipo_y_forma(nombre: str):
    n = nombre.upper()

    if "FL2A" in n:
        return "fluidez", "2A"
    if "FL2B" in n:
        return "fluidez", "2B"
    if "DEC2A" in n:
        return "decodificacion", "2A"
    if "DEC2B" in n:
        return "decodificacion", "2B"

    return None, None

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
        return "mixto"
    if nf == "BAJO":
        return "fluidez"
    if nd == "BAJO":
        return "decodificacion"
    return "sin_dificultad"

def estado_lectura(nf: str | None, nd: str | None):
    if nf == "ALTO" and nd == "ALTO":
        return "consolidado"
    if nf == "BAJO" and nd == "BAJO":
        return "descendido"
    if nf == "MEDIO" or nd == "MEDIO":
        return "en_desarrollo"
    if nf == "BAJO" or nd == "BAJO":
        return "en_riesgo"
    return "indeterminado"

def interpretar_resultado(nf: str | None, nd: str | None):
    if nf == "BAJO" and nd == "BAJO":
        return (
            "El estudiante presenta dificultades tanto en decodificación como en fluidez, "
            "lo que sugiere un proceso lector poco automatizado y con debilidades en precisión y velocidad."
        )
    if nf == "BAJO" and nd in {"MEDIO", "ALTO"}:
        return (
            "El estudiante muestra una dificultad predominante en fluidez. "
            "La decodificación parece relativamente preservada, pero la lectura aún no se automatiza adecuadamente."
        )
    if nd == "BAJO" and nf in {"MEDIO", "ALTO"}:
        return (
            "El estudiante muestra una dificultad predominante en decodificación. "
            "Esto sugiere problemas en reconocimiento preciso de palabras que pueden afectar el rendimiento lector global."
        )
    if nf == "MEDIO" or nd == "MEDIO":
        return (
            "El estudiante se encuentra en una zona intermedia de desarrollo lector. "
            "Se recomienda seguimiento y fortalecimiento focalizado según el componente más descendido."
        )
    if nf == "ALTO" and nd == "ALTO":
        return (
            "El estudiante presenta un desempeño lector consolidado en fluidez y decodificación."
        )
    return "No fue posible generar una interpretación precisa."

def badge_color(nivel: str | None):
    if nivel == "ALTO":
        return "#d1fae5"
    if nivel == "MEDIO":
        return "#fef3c7"
    if nivel == "BAJO":
        return "#fee2e2"
    return "#e5e7eb"

def texto_desde_nombre(nombre: str):
    n = nombre.upper()
    if "MALO" in n:
        return "malo"
    if "BUENO" in n:
        return "bueno"
    return "neutro"

def generar_indices_simulados(nombre_fluidez: str, nombre_decod: str):
    marcador_f = texto_desde_nombre(nombre_fluidez)
    marcador_d = texto_desde_nombre(nombre_decod)

    if marcador_f == "malo":
        fluidez_index = round(random.uniform(0.20, 0.39), 2)
    elif marcador_f == "bueno":
        fluidez_index = round(random.uniform(0.75, 0.95), 2)
    else:
        fluidez_index = round(random.uniform(0.40, 0.74), 2)

    if marcador_d == "malo":
        decod_index = round(random.uniform(0.20, 0.39), 2)
    elif marcador_d == "bueno":
        decod_index = round(random.uniform(0.75, 0.95), 2)
    else:
        decod_index = round(random.uniform(0.40, 0.74), 2)

    return fluidez_index, decod_index

@app.get("/", response_class=HTMLResponse)
@app.head("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>Fluidez App v2</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 860px;
                margin: 40px auto;
                padding: 20px;
                line-height: 1.5;
            }
            .card {
                border: 1px solid #ddd;
                border-radius: 12px;
                padding: 24px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            }
            h1, h2 { margin-top: 0; }
            input, button {
                padding: 10px;
                margin-top: 6px;
                margin-bottom: 16px;
                width: 100%;
                box-sizing: border-box;
            }
            button {
                cursor: pointer;
                font-weight: bold;
            }
            .small {
                color: #555;
                font-size: 14px;
            }
            .demo {
                background: #f3f4f6;
                padding: 12px;
                border-radius: 8px;
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>Fluidez App v2</h1>
            <p>Sube un audio de fluidez y uno de decodificación para obtener un perfil lector integrado.</p>

            <div class="demo">
                Esta versión entrega resultados simulados inteligentes según el nombre del archivo
                (por ejemplo, "MALO" o "BUENO"), mientras se integra el motor real de análisis.
            </div>

            <form action="/evaluar" method="post" enctype="multipart/form-data">
                <label><strong>ID del estudiante</strong></label>
                <input type="text" name="student_id" placeholder="Ej: S001" required />

                <label><strong>Archivo de Fluidez</strong></label>
                <input type="file" name="fluidez_file" required />

                <label><strong>Archivo de Decodificación</strong></label>
                <input type="file" name="decod_file" required />

                <button type="submit">Evaluar</button>
            </form>

            <p class="small">
                El nombre del archivo debe incluir FL2A o FL2B para fluidez, y DEC2A o DEC2B para decodificación.
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
    fluidez_file: UploadFile = File(...),
    decod_file: UploadFile = File(...),
):
    student_id_safe = html.escape(student_id.strip())
    fluidez_name = fluidez_file.filename or "sin_nombre"
    decod_name = decod_file.filename or "sin_nombre"

    validar_extension(fluidez_name)
    validar_extension(decod_name)

    tipo_f, forma_f = detectar_tipo_y_forma(fluidez_name)
    tipo_d, forma_d = detectar_tipo_y_forma(decod_name)

    if tipo_f != "fluidez":
        return """
        <h2>Error</h2>
        <p>El archivo de fluidez no contiene una etiqueta válida como FL2A o FL2B.</p>
        <a href="/">Volver</a>
        """

    if tipo_d != "decodificacion":
        return """
        <h2>Error</h2>
        <p>El archivo de decodificación no contiene una etiqueta válida como DEC2A o DEC2B.</p>
        <a href="/">Volver</a>
        """

    if forma_f != forma_d:
        return f"""
        <h2>Error</h2>
        <p>Las formas no coinciden: Fluidez = {forma_f}, Decodificación = {forma_d}.</p>
        <p>Debes subir archivos de la misma forma (por ejemplo, ambos 2A).</p>
        <a href="/">Volver</a>
        """

    fluidez_path = UPLOAD_DIR / Path(fluidez_name).name
    decod_path = UPLOAD_DIR / Path(decod_name).name

    with open(fluidez_path, "wb") as buffer:
        shutil.copyfileobj(fluidez_file.file, buffer)

    with open(decod_path, "wb") as buffer:
        shutil.copyfileobj(decod_file.file, buffer)

    fluidez_index, decod_index = generar_indices_simulados(fluidez_name, decod_name)

    nivel_f = clasificar(fluidez_index)
    nivel_d = clasificar(decod_index)

    perfil = perfil_integrado(nivel_f, nivel_d)
    problema = tipo_problema(nivel_f, nivel_d)
    estado = estado_lectura(nivel_f, nivel_d)
    interpretacion = interpretar_resultado(nivel_f, nivel_d)

    fluidez_path.unlink(missing_ok=True)
    decod_path.unlink(missing_ok=True)

    fluidez_name_safe = html.escape(fluidez_name)
    decod_name_safe = html.escape(decod_name)

    return f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>Resultado</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 920px;
                margin: 40px auto;
                padding: 20px;
                line-height: 1.5;
            }}
            .card {{
                border: 1px solid #ddd;
                border-radius: 12px;
                padding: 24px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            }}
            .grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 16px;
                margin-top: 20px;
            }}
            .box {{
                border: 1px solid #e5e7eb;
                border-radius: 10px;
                padding: 16px;
            }}
            .badge {{
                display: inline-block;
                padding: 6px 10px;
                border-radius: 999px;
                font-weight: bold;
            }}
            .summary {{
                margin-top: 20px;
                background: #f9fafb;
                border-radius: 10px;
                padding: 16px;
            }}
            a {{
                display: inline-block;
                margin-top: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="card">
            <h1>Resultado de evaluación</h1>
            <p><strong>ID estudiante:</strong> {student_id_safe}</p>
            <p><strong>Forma:</strong> {forma_f}</p>

            <div class="grid">
                <div class="box">
                    <h2>Fluidez</h2>
                    <p><strong>Archivo:</strong> {fluidez_name_safe}</p>
                    <p><strong>Índice:</strong> {fluidez_index}</p>
                    <p>
                        <strong>Nivel:</strong>
                        <span class="badge" style="background:{badge_color(nivel_f)};">{nivel_f}</span>
                    </p>
                </div>

                <div class="box">
                    <h2>Decodificación</h2>
                    <p><strong>Archivo:</strong> {decod_name_safe}</p>
                    <p><strong>Índice:</strong> {decod_index}</p>
                    <p>
                        <strong>Nivel:</strong>
                        <span class="badge" style="background:{badge_color(nivel_d)};">{nivel_d}</span>
                    </p>
                </div>
            </div>

            <div class="summary">
                <p><strong>Perfil integrado:</strong> {perfil}</p>
                <p><strong>Tipo de problema:</strong> {problema}</p>
                <p><strong>Estado lector:</strong> {estado}</p>
                <p><strong>Interpretación:</strong> {interpretacion}</p>
            </div>

            <a href="/">Evaluar otro estudiante</a>
        </div>
    </body>
    </html>
    """
