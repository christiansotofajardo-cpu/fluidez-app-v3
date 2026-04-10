from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from pathlib import Path
import shutil
import re
import wave
import contextlib
import pandas as pd
from rapidfuzz import fuzz
import whisper

app = FastAPI(title="Fluidez App v3")

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_NAME = "tiny"
model = whisper.load_model(MODEL_NAME)

WHISPER_KW = dict(
    language="es",
    temperature=0,
    condition_on_previous_text=False,
    fp16=False
)

DEFAULT_FL_2A = """una mañana el duende melodía decidió limpiar su desván donde había guardado cosas por más de cien años puso una escalera trepó al techo de su casa y abrió una pequeña puerta saludó a la araña que cuidaba el desván comiéndose las polillas cómo está doña pepa vengo a ordenar un poco yo tengo todo en orden contestó pepa columpiándose en su tela ya sé que a usted no le gusta que trajine por aquí pero lo haré sin espantar ni una polilla gracias no me gusta quedarme sin almuerzo contestó la araña arreglándose el gorro de piel no pasaron dos segundos cuando se oyó una voz muy conocida oye duende yo quiero subir al desván"""

DEFAULT_FL_2B = """la vieja hormiga vivía en el fondo de un inmenso hormiguero estaba casi ciega y sorda pero era más sabia que la reina de las hormigas un día cuando empezaba la primavera la vieja sintió golpes de muchas patas que corrían por los pasillos y galerías sobre su cabeza ese pataleo significa que pronto despertarán las hormigas nuevas murmuró la anciana apoyándose en un bastón de paja subió de galería en galería hasta llegar a la sala cuna ahí las hormigas nuevas las crisálidas dormían aún envueltas en sus chales blancos la mayordoma vigilaba para que todo estuviera limpio y preparado para el momento del despertar"""

DEC_2A = [
    "indignado", "cobujango", "grabadora", "indestructible", "prectun", "intercambiar", "asderminotar", "inventor",
    "flesujas", "pantelirdo", "independencia", "gracioso", "tosperantago", "lornifaru", "actuar", "resajes"
]

DEC_2B = [
    "adorable", "olpictaver", "publicidad", "internacional", "placter", "diferenciar", "turotenacer", "columna",
    "rintumos", "trolastaña", "intranquilidad", "valiente", "custorguljera", "mucapisa", "trepar", "taloncer"
]

WCPM_MAX_TEO = 120.0


def normalize_text(s: str) -> str:
    s = (s or "").lower()
    repl = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ü": "u", "ñ": "n"}
    for k, v in repl.items():
        s = s.replace(k, v)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(s: str) -> list[str]:
    return normalize_text(s).split()


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


def base_id(nombre: str) -> str:
    return re.sub(r"_(FL|DEC)2[A-B]\.wav$", "", nombre, flags=re.IGNORECASE)


def transcribe(path: Path) -> str:
    try:
        r = model.transcribe(str(path), **WHISPER_KW)
        return (r.get("text") or "").strip()
    except Exception as e:
        return f"ERROR_TRANSCRIPCION: {e}"


def get_wav_duration(path: Path) -> float:
    try:
        with contextlib.closing(wave.open(str(path), "rb")) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            if rate == 0:
                return 0.0
            return round(frames / float(rate), 2)
    except Exception:
        return 0.0


def levenshtein_ops(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    op = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        op[i][0] = "del"
    for j in range(1, m + 1):
        dp[0][j] = j
        op[0][j] = "ins"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            choices = [
                (dp[i - 1][j] + 1, "del"),
                (dp[i][j - 1] + 1, "ins"),
                (dp[i - 1][j - 1] + cost, "sub" if cost else "ok"),
            ]
            dp[i][j], op[i][j] = min(choices, key=lambda x: x[0])

    i, j = n, m
    correct = subs = ins = dele = 0
    while i > 0 or j > 0:
        o = op[i][j]
        if o in ("ok", "sub"):
            if o == "ok":
                correct += 1
            else:
                subs += 1
            i -= 1
            j -= 1
        elif o == "del":
            dele += 1
            i -= 1
        else:
            ins += 1
            j -= 1

    return correct, subs, dele, ins


def best_match(expected, tokens, start_idx, window=8):
    best_score = 0
    best_next = start_idx
    expn = normalize_text(expected)
    end = min(len(tokens), start_idx + window)

    for i in range(start_idx, end):
        score1 = fuzz.ratio(expn, tokens[i])
        if score1 > best_score:
            best_score = score1
            best_next = i + 1

        if i + 1 < end:
            comb = tokens[i] + tokens[i + 1]
            score2 = fuzz.ratio(expn, comb)
            if score2 > best_score:
                best_score = score2
                best_next = i + 2

    return best_score, best_next


def clamp01(x):
    try:
        x = float(x)
    except Exception:
        return pd.NA
    if x < 0:
        return 0.0
    if x > 1:
        return 1.0
    return round(x, 4)


def clasificar_indice(indice):
    if pd.isna(indice):
        return pd.NA
    if indice >= 0.75:
        return "ALTO"
    if indice >= 0.40:
        return "MEDIO"
    return "BAJO"


def comentario_fluidez(nivel):
    if nivel == "ALTO":
        return "Lectura fluida y automatizada; alta precisión, continuidad y eficiencia."
    if nivel == "MEDIO":
        return "Fluidez intermedia; hay avances, pero aún se observan quiebres de continuidad o precisión."
    if nivel == "BAJO":
        return "Fluidez baja; se observan dificultades importantes de precisión y/o continuidad."
    return pd.NA


def comentario_decod(nivel):
    if nivel == "ALTO":
        return "Decodificación sólida; buen reconocimiento de palabras y pseudopalabras."
    if nivel == "MEDIO":
        return "Decodificación intermedia; requiere refuerzo parcial."
    if nivel == "BAJO":
        return "Decodificación baja; se observan dificultades importantes de reconocimiento fonológico-ortográfico."
    return pd.NA


def perfil_integrado(nf, nd):
    if pd.notna(nf) and pd.notna(nd):
        return f"Fluidez {nf} - Decodificación {nd}"
    if pd.notna(nf):
        return f"Solo Fluidez {nf}"
    if pd.notna(nd):
        return f"Solo Decodificación {nd}"
    return "Sin datos"


def tipo_problema_refinado(nf, nd):
    if pd.notna(nf) and pd.notna(nd):
        if nf == "ALTO" and nd == "ALTO":
            return "sin_dificultad"
        if nf == "ALTO" and nd == "MEDIO":
            return "en_transicion"
        if nf == "MEDIO" and nd == "ALTO":
            return "automatizacion"
        if nf == "MEDIO" and nd == "MEDIO":
            return "en_desarrollo"
        if nf == "BAJO" and nd == "BAJO":
            return "mixto"
        if nd == "BAJO" and nf != "BAJO":
            return "decodificacion"
        if nf == "BAJO" and nd != "BAJO":
            return "automatizacion_severa"
        return "indeterminado"
    if pd.notna(nf):
        return "solo_fluidez"
    if pd.notna(nd):
        return "solo_decodificacion"
    return "sin_datos"


def estado_lectura(nf, nd):
    if pd.notna(nf) and pd.notna(nd):
        if nf == "ALTO" and nd == "ALTO":
            return "consolidado"
        if nf == "ALTO" and nd == "MEDIO":
            return "casi_consolidado"
        if nf == "MEDIO" and nd == "ALTO":
            return "automatizacion_incompleta"
        if nf == "MEDIO" and nd == "MEDIO":
            return "en_desarrollo"
        if nf == "BAJO" and nd == "BAJO":
            return "descendido"
        if nd == "BAJO":
            return "base_fragil"
        if nf == "BAJO":
            return "fluidez_descendida"
        return "indeterminado"
    if pd.notna(nf):
        return "solo_fluidez"
    if pd.notna(nd):
        return "solo_decodificacion"
    return "sin_datos"


def generar_excel(df_integrado, df_fluidez, df_decod, df_debug, output_path: Path):
    df_resumen = pd.DataFrame([{
        "n_total_filas": len(df_integrado),
        "solo_fluidez": int((df_integrado["perfil_integrado"].astype(str).str.contains("Solo Fluidez", na=False)).sum()) if "perfil_integrado" in df_integrado.columns else 0,
        "solo_decodificacion": int((df_integrado["perfil_integrado"].astype(str).str.contains("Solo Decodificación", na=False)).sum()) if "perfil_integrado" in df_integrado.columns else 0,
        "integrados": int((df_integrado["perfil_integrado"].astype(str).str.contains("Fluidez", na=False) & df_integrado["perfil_integrado"].astype(str).str.contains("Decodificación", na=False)).sum()) if "perfil_integrado" in df_integrado.columns else 0,
    }])

    definiciones = [
        ["VALORES BRUTOS", "Valores observados directamente en la tarea, antes de cualquier transformación."],
        ["VALORES NORMALIZADOS", "Escalas provisionales 0-1 basadas en máximos teóricos de tarea, no en corpus empírico."],
        ["fluidez_index_norm", "Índice sintético de fluidez basado en variables normalizadas y ponderación refinada."],
        ["decod_index_norm", "Índice sintético de decodificación basado en variables normalizadas."],
        ["lectura_global_index_norm", "Promedio de fluidez_index_norm y decod_index_norm cuando ambas medidas existen."],
        ["perfil_integrado", "Síntesis cualitativa que integra fluidez y decodificación."],
        ["tipo_problema", "Clasificación orientativa refinada del tipo de dificultad dominante."],
        ["estado_lectura", "Estado interpretativo del sistema lector."],
        ["NOTA METODOLÓGICA", "Las normalizaciones de esta versión son provisionales y teóricas; podrán refinarse con corpus empírico."],
    ]
    df_def = pd.DataFrame(definiciones, columns=["indice", "definicion"])

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_integrado.to_excel(writer, index=False, sheet_name="integrado")
        df_fluidez.to_excel(writer, index=False, sheet_name="solo_fluidez")
        df_decod.to_excel(writer, index=False, sheet_name="solo_decod")
        df_resumen.to_excel(writer, index=False, sheet_name="resumen")
        df_debug.to_excel(writer, index=False, sheet_name="debug")
        df_def.to_excel(writer, index=False, sheet_name="definiciones")


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <title>Fluidez App v3</title>
    </head>
    <body>
        <h1>Fluidez App v3</h1>
        <p>Sube un audio .wav con nombre tipo FL2A, FL2B, DEC2A o DEC2B.</p>
        <form action="/upload" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept=".wav"/>
            <button type="submit">Subir y procesar</button>
        </form>
    </body>
    </html>
    """


@app.get("/download/{filename}")
def download_file(filename: str):
    file_path = OUTPUT_DIR / filename
    if file_path.exists():
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    return HTMLResponse("<h3>Archivo no encontrado.</h3>", status_code=404)


@app.post("/upload", response_class=HTMLResponse)
async def upload(file: UploadFile = File(...)):
    filename = file.filename or "audio.wav"
    filepath = UPLOAD_DIR / filename

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    tipo, forma = detectar_tipo_y_forma(filename)
    if tipo is None:
        return HTMLResponse(
            """
            <h2>Archivo recibido, pero el nombre no permite identificar la tarea.</h2>
            <p>Usa nombres con FL2A, FL2B, DEC2A o DEC2B.</p>
            <p><a href="/">Volver</a></p>
            """,
            status_code=200
        )

    dur = get_wav_duration(filepath)
    txt = transcribe(filepath)
    tokens = tokenize(txt)

    rows_debug = [{
        "archivo": filename,
        "base": base_id(filename),
        "tipo": tipo,
        "forma": forma,
        "duracion_seg": dur,
        "n_tokens_transcritos": len(tokens),
        "transcripcion": txt,
    }]

    rows_fluidez = []
    rows_decod = []

    if tipo == "fluidez":
        esperado = DEFAULT_FL_2A if forma == "2A" else DEFAULT_FL_2B
        exp_tokens = tokenize(esperado)

        correct, subs, dels, ins = levenshtein_ops(exp_tokens, tokens)
        words_expected = len(exp_tokens)
        accuracy = round(100 * correct / words_expected, 1) if words_expected else 0
        wcpm = round(correct / (dur / 60.0), 1) if dur > 0 else 0

        correct_norm = clamp01(correct / words_expected) if words_expected else pd.NA
        omissions_norm = clamp01(1 - (dels / words_expected)) if words_expected else pd.NA
        substitutions_norm = clamp01(1 - (subs / words_expected)) if words_expected else pd.NA
        insertions_norm = clamp01(1 - (ins / words_expected)) if words_expected else pd.NA
        accuracy_norm = clamp01(accuracy / 100)
        wcpm_norm = clamp01(wcpm / WCPM_MAX_TEO)

        weighted_values = []
        weighted_weights = []
        for value, weight in [
            (correct_norm, 0.35),
            (omissions_norm, 0.25),
            (substitutions_norm, 0.20),
            (wcpm_norm, 0.20),
        ]:
            if pd.notna(value):
                weighted_values.append(value * weight)
                weighted_weights.append(weight)

        fluidez_index_norm = round(sum(weighted_values) / sum(weighted_weights), 4) if weighted_weights else pd.NA
        nivel_fluidez = clasificar_indice(fluidez_index_norm)

        rows_fluidez.append({
            "base": base_id(filename),
            "forma": forma,
            "archivo_fluidez": filename,
            "duracion_fluidez_seg": dur,
            "words_expected_bruto": words_expected,
            "tokens_fluidez_bruto": len(tokens),
            "correct_words_bruto": correct,
            "substitutions_bruto": subs,
            "omissions_bruto": dels,
            "insertions_bruto": ins,
            "accuracy_percent_bruto": accuracy,
            "WCPM_bruto": wcpm,
            "correct_words_norm": correct_norm,
            "substitutions_norm": substitutions_norm,
            "omissions_norm": omissions_norm,
            "insertions_norm": insertions_norm,
            "accuracy_percent_norm": accuracy_norm,
            "WCPM_norm": wcpm_norm,
            "fluidez_index_norm": fluidez_index_norm,
            "nivel_fluidez": nivel_fluidez,
            "comentario_fluidez": comentario_fluidez(nivel_fluidez),
            "transcripcion_fluidez": txt,
        })

        df_fluidez = pd.DataFrame(rows_fluidez)
        df_decod = pd.DataFrame(columns=[
            "base", "forma", "archivo_decod", "duracion_decod_seg", "items_totales_bruto",
            "tokens_decod_bruto", "items_correctos_bruto", "accuracy_decod_percent_bruto",
            "items_correctos_norm", "accuracy_decod_percent_norm", "decod_index_norm",
            "nivel_decod", "comentario_decod", "transcripcion_decod"
        ])
        df_integrado = df_fluidez.copy()
        for col in [
            "archivo_decod", "duracion_decod_seg", "items_totales_bruto", "tokens_decod_bruto",
            "items_correctos_bruto", "accuracy_decod_percent_bruto", "items_correctos_norm",
            "accuracy_decod_percent_norm", "decod_index_norm", "nivel_decod", "comentario_decod",
            "transcripcion_decod"
        ]:
            df_integrado[col] = pd.NA

    else:
        esperadas = DEC_2A if forma == "2A" else DEC_2B
        idx = 0
        correct = 0
        for exp in esperadas:
            score, idx = best_match(exp, tokens, idx, window=8)
            if score >= 80:
                correct += 1

        total_items = len(esperadas)
        accuracy = round(100 * correct / total_items, 1) if total_items else 0

        items_correctos_norm = clamp01(correct / total_items) if total_items else pd.NA
        accuracy_decod_norm = clamp01(accuracy / 100)
        dec_components = [x for x in [items_correctos_norm, accuracy_decod_norm] if pd.notna(x)]
        decod_index_norm = round(sum(dec_components) / len(dec_components), 4) if dec_components else pd.NA
        nivel_decod = clasificar_indice(decod_index_norm)

        rows_decod.append({
            "base": base_id(filename),
            "forma": forma,
            "archivo_decod": filename,
            "duracion_decod_seg": dur,
            "items_totales_bruto": total_items,
            "tokens_decod_bruto": len(tokens),
            "items_correctos_bruto": correct,
            "accuracy_decod_percent_bruto": accuracy,
            "items_correctos_norm": items_correctos_norm,
            "accuracy_decod_percent_norm": accuracy_decod_norm,
            "decod_index_norm": decod_index_norm,
            "nivel_decod": nivel_decod,
            "comentario_decod": comentario_decod(nivel_decod),
            "transcripcion_decod": txt,
        })

        df_decod = pd.DataFrame(rows_decod)
        df_fluidez = pd.DataFrame(columns=[
            "base", "forma", "archivo_fluidez", "duracion_fluidez_seg", "words_expected_bruto",
            "tokens_fluidez_bruto", "correct_words_bruto", "substitutions_bruto",
            "omissions_bruto", "insertions_bruto", "accuracy_percent_bruto", "WCPM_bruto",
            "correct_words_norm", "substitutions_norm", "omissions_norm", "insertions_norm",
            "accuracy_percent_norm", "WCPM_norm", "fluidez_index_norm", "nivel_fluidez",
            "comentario_fluidez", "transcripcion_fluidez"
        ])
        df_integrado = df_decod.copy()
        for col in [
            "archivo_fluidez", "duracion_fluidez_seg", "words_expected_bruto", "tokens_fluidez_bruto",
            "correct_words_bruto", "substitutions_bruto", "omissions_bruto", "insertions_bruto",
            "accuracy_percent_bruto", "WCPM_bruto", "correct_words_norm", "substitutions_norm",
            "omissions_norm", "insertions_norm", "accuracy_percent_norm", "WCPM_norm",
            "fluidez_index_norm", "nivel_fluidez", "comentario_fluidez", "transcripcion_fluidez"
        ]:
            df_integrado[col] = pd.NA

    def lectura_global_index(row):
        fi = row.get("fluidez_index_norm")
        di = row.get("decod_index_norm")
        vals = [x for x in [fi, di] if pd.notna(x)]
        return round(sum(vals) / len(vals), 4) if vals else pd.NA

    df_integrado["perfil_integrado"] = df_integrado.apply(
        lambda r: perfil_integrado(r.get("nivel_fluidez"), r.get("nivel_decod")), axis=1
    )
    df_integrado["tipo_problema"] = df_integrado.apply(
        lambda r: tipo_problema_refinado(r.get("nivel_fluidez"), r.get("nivel_decod")), axis=1
    )
    df_integrado["estado_lectura"] = df_integrado.apply(
        lambda r: estado_lectura(r.get("nivel_fluidez"), r.get("nivel_decod")), axis=1
    )
    df_integrado["lectura_global_index_norm"] = df_integrado.apply(lectura_global_index, axis=1)

    df_debug = pd.DataFrame(rows_debug)

    excel_name = f"resultado_{filepath.stem}.xlsx"
    excel_path = OUTPUT_DIR / excel_name
    generar_excel(df_integrado, df_fluidez, df_decod, df_debug, excel_path)

    nivel_fluidez = df_integrado.iloc[0].get("nivel_fluidez", pd.NA)
    nivel_decod = df_integrado.iloc[0].get("nivel_decod", pd.NA)
    perfil = df_integrado.iloc[0].get("perfil_integrado", "Sin datos")
    tipo_prob = df_integrado.iloc[0].get("tipo_problema", "sin_datos")
    estado = df_integrado.iloc[0].get("estado_lectura", "sin_datos")
    global_index = df_integrado.iloc[0].get("lectura_global_index_norm", pd.NA)

    return f"""
    <html>
    <head><title>Resultado</title></head>
    <body>
        <h1>Resultado procesado</h1>
        <p><strong>Archivo:</strong> {filename}</p>
        <p><strong>Tipo detectado:</strong> {tipo}</p>
        <p><strong>Forma:</strong> {forma}</p>
        <p><strong>Duración (seg):</strong> {dur}</p>
        <p><strong>Nivel fluidez:</strong> {nivel_fluidez}</p>
        <p><strong>Nivel decodificación:</strong> {nivel_decod}</p>
        <p><strong>Perfil integrado:</strong> {perfil}</p>
        <p><strong>Tipo de problema:</strong> {tipo_prob}</p>
        <p><strong>Estado lectura:</strong> {estado}</p>
        <p><strong>Índice global:</strong> {global_index}</p>
        <p><strong>Transcripción:</strong> {txt}</p>
        <p><a href="/download/{excel_name}">Descargar Excel</a></p>
        <p><a href="/">Volver</a></p>
    </body>
    </html>
    ```
    
Luego:
1. guarda con **Commit changes**
2. vuelve a Render
3. debería redeployar solo

Cuando termine, sube de nuevo un audio y me dices qué te mostró.
