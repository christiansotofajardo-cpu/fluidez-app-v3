from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from pathlib import Path
import shutil
import random
import html

app = FastAPI(title="Fluidez App")

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
    # Caso con ambos datos
    if nf and nd:
        if nf == "ALTO" and nd == "ALTO":
            return "consolidado"
        if nf == "MEDIO" or nd == "MEDIO":
            return "en_desarrollo"
        if nf == "BAJO" or nd == "BAJO":
            return "descendido"
        return "indeterminado"

    # Caso parcial
    if nf or nd:
        nivel = nf or nd
        if nivel == "ALTO":
            return "adecuado_parcial"
        if nivel == "MEDIO":
            return "en_desarrollo_parcial"
        if nivel == "BAJO":
            return "descendido_parcial"

    return "sin_datos"

def validar_archivo(filename: str):
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Extensión no permitida. Usa: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

@app.get("/", response_class=HTMLResponse)
@app.head("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>Fluidez App</title>
    </head>
    <body>
        <h1>Fluidez App (Demo)</h1>
        <p>Sube un archivo de audio con nombre tipo FL2A, FL2B, DEC2A o DEC2B.</p>
        <form action="/upload" method="post" enctype="multipart/form-data">
            <input type="file" name="file" required />
            <button type="submit">Subir</button>
        </form>
    </body>
    </html>
    """

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return HTMLResponse(status_code=204, content="")

@app.post("/upload", response_class=HTMLResponse)
async def upload(file: UploadFile = File(...)):
    safe_name = html.escape(file.filename or "archivo_sin_nombre")
    validar_archivo(file.filename or "")

    filepath = UPLOAD_DIR / Path(file.filename).name

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    tipo, forma = detectar_tipo_y_forma(file.filename or "")
    if tipo is None:
        filepath.unlink(missing_ok=True)
        return """
        <h2>No se reconoce el tipo de archivo.</h2>
        <p>El nombre debe contener FL2A, FL2B, DEC2A o DEC2B.</p>
        <a href="/">Volver</a>
        """

    # DEMO: simulación temporal
    fluidez_index = round(random.uniform(0.3, 0.9), 2) if tipo == "fluidez" else None
    decod_index = round(random.uniform(0.3, 0.9), 2) if tipo == "decodificacion" else None

    nivel_f = clasificar(fluidez_index)
    nivel_d = clasificar(decod_index)

    perfil = perfil_integrado(nivel_f, nivel_d)
    problema = tipo_problema(nivel_f, nivel_d)
    estado = estado_lectura(nivel_f, nivel_d)

    # Opcional: borrar archivo tras procesarlo en esta versión demo
    filepath.unlink(missing_ok=True)

    return f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>Resultado</title>
    </head>
    <body>
        <h1>Resultado</h1>
        <p><strong>Archivo:</strong> {safe_name}</p>
        <p><strong>Tipo:</strong> {tipo}</p>
        <p><strong>Forma:</strong> {forma}</p>
        <p><strong>Índice Fluidez:</strong> {fluidez_index if fluidez_index is not None else "No aplica"}</p>
        <p><strong>Índice Decodificación:</strong> {decod_index if decod_index is not None else "No aplica"}</p>
        <p><strong>Nivel Fluidez:</strong> {nivel_f or "No aplica"}</p>
        <p><strong>Nivel Decodificación:</strong> {nivel_d or "No aplica"}</p>
        <p><strong>Perfil:</strong> {perfil}</p>
        <p><strong>Tipo de problema:</strong> {problema}</p>
        <p><strong>Estado lector:</strong> {estado}</p>
        <br>
        <a href="/">Subir otro archivo</a>
    </body>
    </html>
    """
