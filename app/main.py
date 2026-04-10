from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import shutil
from pathlib import Path

app = FastAPI()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h2>Fluidez App v3</h2>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <input type="file" name="file"/>
        <button type="submit">Subir</button>
    </form>
    """

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    filepath = UPLOAD_DIR / file.filename
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {"filename": file.filename, "status": "ok"}
