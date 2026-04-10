from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from pathlib import Path
import shutil
import random

app = FastAPI()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

def detectar_tipo_y_forma(nombre):
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

def clasificar(valor):
    if valor >= 0.75:
        return "ALTO"
    elif valor >= 0.40:
        return "MEDIO"
    return "BAJO"

def perfil_integrado(nf, nd):
    if nf and nd:
        return f"Fluidez {nf} - Decodificación {nd}"
    if nf:
        return f"Solo Fluidez {nf}"
    if nd:
        return f"Solo Decodificación {nd}"
    return "Sin datos"

def tipo_problema(nf, nd):
    if nf == "BAJO" and nd == "BAJO":
        return "mixto"
    if nf == "BAJO":
        return "fluidez"
    if nd == "BAJO":
        return "decodificacion"
    return "sin_dificultad"

def estado_lectura(nf, nd):
    if nf == "ALTO" and nd == "ALTO":
        return "consolidado"
    if nf == "MEDIO" or nd == "MEDIO":
        return "en_desarrollo"
    return "descendido"

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h1>Fluidez App (Demo)</h1>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <input type="file" name="file"/>
        <button type="submit">Subir</button>
    </form>
    """

@app.post("/upload", response_class=HTMLResponse)
async def upload(file: UploadFile = File(...)):
    filepath = UPLOAD_DIR / file.filename
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    tipo, forma = detectar_tipo_y_forma(file.filename)

    if tipo is None:
        return "<h2>No se reconoce el tipo (FL2A, DEC2A, etc.)</h2>"

    # SIMULACIÓN INTELIGENTE
    fluidez_index = round(random.uniform(0.3, 0.9), 2) if tipo == "fluidez" else None
    decod_index = round(random.uniform(0.3, 0.9), 2) if tipo == "decodificacion" else None

    nivel_f = clasificar(fluidez_index) if fluidez_index else None
    nivel_d = clasificar(decod_index) if decod_index else None

    perfil = perfil_integrado(nivel_f, nivel_d)
    problema = tipo_problema(nivel_f, nivel_d)
    estado = estado_lectura(nivel_f, nivel_d)

    return f"""
    <h1>Resultado</h1>
    <p><strong>Archivo:</strong> {file.filename}</p>
    <p><strong>Tipo:</strong> {tipo}</p>
    <p><strong>Forma:</strong> {forma}</p>
    <p><strong>Índice Fluidez:</strong> {fluidez_index}</p>
    <p><strong>Índice Decodificación:</strong> {decod_index}</p>
    <p><strong>Nivel Fluidez:</strong> {nivel_f}</p>
    <p><strong>Nivel Decodificación:</strong> {nivel_d}</p>
    <p><strong>Perfil:</strong> {perfil}</p>
    <p><strong>Tipo de problema:</strong> {problema}</p>
    <p><strong>Estado lector:</strong> {estado}</p>
    """
