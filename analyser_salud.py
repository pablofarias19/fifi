# -*- coding: utf-8 -*-
"""
ANALYSER_SALUD (Gemini Pro + RAG sanitario multilingüe) con salida PDF
----------------------------------------------------------------------
- Bases RAG: Salud - Médica, Salud - Laboral, Jurisprudencia - Salud, Legislación - Salud
- LLM: Gemini 2.5 Pro (langchain-google-genai)
- Recuperación: Chroma (+ re-score jurídico-sanitario)
- Entrada: PDF (con OCR opcional en ingesta) o texto/consulta
- Salida: Informe PDF + JSON (stdout)

Uso CLI:
  python analyser_salud.py --pdf ./ejemplo.pdf --tipo "Salud - Médica" --out reporte.pdf
  python analyser_salud.py --query "consentimiento informado en cirugía" --out reporte.pdf
  python analyser_salud.py --text "./texto.txt" --tipo "Jurisprudencia - Salud" --out reporte.pdf
"""

from __future__ import annotations
import os, re, json, math, logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pathlib import Path

# --- LLM (Gemini) ---
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Vectorstore / Embeddings ---
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings as SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# --- PDF Report ---
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

# --- Extracción PDF (reutiliza tu pipeline de ingesta) ---
try:
    from ingesta import extract_text_from_pdf, Settings as SettingsIngesta
except Exception:
    extract_text_from_pdf = None
    SettingsIngesta = None

# ============================================================
# DETECTOR AUTOMÁTICO DE EMBEDDINGS
# ============================================================

def _make_embedder(auto_path: Path = None):
    """
    Crea el modelo de embeddings adecuado según la base detectada.
    - Si no hay base o no se puede leer, usa el modelo estándar (MiniLM-L6-v2, 384d).
    - Si la base contiene vectores 768d, usa all-mpnet-base-v2 automáticamente.
    """
    default_model = "sentence-transformers/all-MiniLM-L6-v2"  # 384d
    advanced_model = "sentence-transformers/all-mpnet-base-v2"  # 768d

    try:
        if auto_path and (auto_path / "chroma.sqlite3").exists():
            # Detección de dimensiones a partir de metadata
            import sqlite3
            conn = sqlite3.connect(auto_path / "chroma.sqlite3")
            cur = conn.cursor()
            cur.execute("SELECT value FROM embedding_metadata LIMIT 1;")
            val = cur.fetchone()
            conn.close()

            # Intentar detectar la longitud del vector
            if val and len(str(val)) > 600:
                print(f"🧠 Base {auto_path.name} detectada como 768d → usando {advanced_model}")
                return SentenceTransformerEmbeddings(model_name=advanced_model)
            else:
                print(f"🧠 Base {auto_path.name} detectada como 384d → usando {default_model}")
                return SentenceTransformerEmbeddings(model_name=default_model)
    except Exception:
        pass

    print(f"⚙️ Sin base previa → usando modelo por defecto ({default_model})")
    return SentenceTransformerEmbeddings(model_name=default_model)

# ======================
# Configuración
# ======================

LOG = logging.getLogger("analyser_salud")
if not LOG.handlers:
    LOG.setLevel(logging.INFO)
    LOG.addHandler(logging.StreamHandler())

ROOT = Path(__file__).parent
DB_DIR_DEFAULT = ROOT / "chroma_db_legal"

BASES_RAG: Dict[str, str] = {
    "Salud - Médica": "pdfs_salud_medica",
    "Salud - Laboral": "pdfs_salud_laboral",
    "Jurisprudencia - Salud": "pdfs_jurisprud_salud",
    "Legislación - Salud": "pdfs_ley_salud",
}

DEFAULT_EMBEDDINGS = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_LLM = "gemini-2.5-pro"



# ======================
# MODELO DE EMBEDDINGS (PARCHE ANTI-META TENSOR)
# ======================
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import SentenceTransformerEmbeddings

def _make_embeddings(model_name: str = DEFAULT_EMBEDDINGS):
    import torch, gc

    # 🔧 Limpieza preventiva de GPU/CPU
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 🔒 Fijamos CPU como destino por defecto (evita meta tensors)
    torch.set_default_device("cpu")

    # 🧠 Cargamos el modelo forzando pre-carga completa de pesos
    model = SentenceTransformer(model_name)
    for name, param in model.named_parameters():
        if param.data.is_meta:
            # 👇 Esta línea crea un tensor real con los datos necesarios
            param.data = torch.zeros_like(param, device="cpu", dtype=param.dtype)
    model.to("cpu")  # Mover con datos reales ya cargados

    # 🧱 Envolvemos el modelo para LangChain
    return SentenceTransformerEmbeddings(client=model)

def _open_vs(persist_dir: Path, collection_name: str = "legal_fragments",
             model_name: str = DEFAULT_EMBEDDINGS) -> Optional[Chroma]:
    if not persist_dir.exists():
        LOG.warning(f"[Chroma] No existe la carpeta de la base: {persist_dir}")
        return None
    embs = _make_embeddings(model_name)
    return Chroma(collection_name=collection_name, embedding_function=embs, persist_directory=str(persist_dir))

def _load_all_retrievers(db_root: Path, bases: Dict[str,str]) -> Dict[str, Chroma]:
    out = {}
    for etiqueta, carpeta in bases.items():
        vs = _open_vs(db_root / carpeta)
        if vs:
            out[etiqueta] = vs
    return out

# ======================
# Búsqueda mejorada (jurídico-sanitaria)
# ======================

_KEYWORDS_JURIDICO_SANIT = (
    "artículo", "articulo", "considerando", "resuelve", "sentencia", "fallo",
    "tribunal", "cámara", "juzgado", "demanda", "dictamen",
    "consentimiento informado", "historia clínica", "lex artis", "mala praxis",
    "paciente", "responsabilidad", "causalidad", "daño", "negligencia",
    "protocolo", "bioética", "OMS", "OPS"
)

def _score_dinamico(query: str, doc: Document, score_sim: float) -> float:
    base = float(score_sim)
    text = doc.page_content.lower()
    kw_hits = sum(1 for k in _KEYWORDS_JURIDICO_SANIT if k in text)
    base += min(0.20, kw_hits * 0.02)
    tipo = doc.metadata.get("tipo_estructura", "parrafo")
    q = query.lower()
    if "considerando" in q and tipo == "considerando": base += 0.05
    if ("art." in q or "artículo" in q or "articulo" in q) and tipo == "articulo": base += 0.05
    if "resuelve" in q and tipo == "resolutivo": base += 0.05
    n = len(doc.page_content.split())
    if 70 <= n <= 900: base += 0.05
    base += min(0.1, float(doc.metadata.get("score_calidad", 0.5)) * 0.1)
    return max(0.0, min(1.5, base))

def enhanced_similarity_search(vs: Chroma, query: str, k: int = 8, fetch_k: int = 32) -> List[Tuple[Document, float]]:
    try:
        base_docs = vs.similarity_search_with_score(query, k=fetch_k)
    except Exception:
        docs = vs.similarity_search(query, k=fetch_k)
        base_docs = [(d, 0.5) for d in docs]
    rescored = [(doc, _score_dinamico(query, doc, score)) for (doc, score) in base_docs]
    rescored.sort(key=lambda x: x[1], reverse=True)
    return rescored[:k]

# ======================
# Métricas vectoriales (auditoría rápida)
# ======================

@dataclass
class VectorAudit:
    base: str
    frags: int
    avg_words: float
    diversity: float
    coverage_types: float
    rating: float

def _audit_vectorstore(vs: Chroma, nombre: str, sample: int = 400) -> VectorAudit:
    try:
        docs = vs.similarity_search("", k=sample)
    except Exception:
        docs = []
    if not docs:
        return VectorAudit(nombre, 0, 0.0, 0.0, 0.0, 0.0)
    sizes = [len(d.page_content.split()) for d in docs]
    avgw = sum(sizes)/len(sizes)
    vocab = set()
    tot = 0
    for d in docs:
        ws = d.page_content.lower().split()
        tot += len(ws)
        vocab.update(ws)
    diversity = min(1.0, (len(vocab)/max(tot,1))*3)
    tipos = [d.metadata.get("tipo_estructura","parrafo") for d in docs]
    coverage = min(1.0, len(set(tipos))/5.0)
    rating = round((min(1.0,avgw/400.0) + diversity + coverage)/3 * 5, 2)
    return VectorAudit(nombre, len(docs), avgw, diversity, coverage, rating)

# ======================
# Prompts (jurídico-médicos, multilingües)
# ======================

SYSTEM_INSTRUCCIONES = (
    "Eres un analista jurídico-médico multilingüe (ES/EN/FR/DE/IT). "
    "Evalúas responsabilidad profesional médica, lex artis, consentimiento informado, "
    "causalidad, daño, nexo adecuado y estándares OMS/OPS. Estructura tus respuestas."
)

PROMPT_ANALISIS = """
[Contexto recuperado]
{contexto}

[Texto/Consulta]
{texto}

[Instrucciones]
1) Identifica los ejes: consentimiento informado, lex artis, causalidad adecuada, daños reclamados, defensa probable.
2) Cita normas y jurisprudencia relevantes (preferir fragmentos recuperados).
3) Señala debilidades probatorias y qué prueba falta.
4) Expón riesgos y probabilidad de éxito (baja/media/alta) con justificación.
5) Si el texto no es claramente médico, delimita alcance sanitario.

[Salida requerida - JSON]
Devuelve un JSON con:
- "tesis": resumen conclusivo
- "fundamentos": lista de fundamentos (normas, fallos, doctrina)
- "riesgos": lista de riesgos
- "prueba_faltante": lista
- "probabilidad_exito": "baja|media|alta"
- "notas": observaciones adicionales
"""

## ======================
# CONTEXTO RAG CON TRAZABILIDAD DOCUMENTAL AVANZADA
## ======================

def _build_context(query_or_text: str, retrievers: Dict[str, Chroma], k_por_base: int = 5) -> str:
    """
    Recupera fragmentos de las bases vectoriales con metadatos enriquecidos:
    autor, título, año, jurisdicción, página, URL, tipo.
    Genera un contexto completo y legible para Gemini.
    """
    bloques = []
    for etiqueta, vs in retrievers.items():
        try:
            emb = _make_embedder(Path(vs._persist_directory))
            results = vs.similarity_search_with_score(query_or_text, k=k_por_base, embedding_function=emb)
            for d, _ in results:
                meta = d.metadata or {}

                fuente = meta.get("archivo_origen", "?")
                autor = meta.get("autor", "Autor desconocido")
                titulo = meta.get("titulo", fuente)
                pagina = meta.get("pagina", meta.get("page", "s/p"))
                jurisdiccion = meta.get("jurisdiccion", "Desconocida")
                idioma = meta.get("idioma", "ES")
                tipo = meta.get("tipo", "texto")
                anio = meta.get("anio", "s/f")
                url = meta.get("url", "")
                origen = meta.get("origen", etiqueta)

                ref = f"{autor} – *{titulo}* ({anio}), pág. {pagina}"
                if url:
                    ref += f" [{url}]"

                encabezado = (
                    f"📚 FUENTE: {ref} | Jurisdicción: {jurisdiccion} | "
                    f"Idioma: {idioma} | Tipo: {tipo} | Base: {origen}"
                )

                contenido = d.page_content.strip().replace("\n", " ")
                resumen = contenido[:1200]
                bloques.append(f"{encabezado}\n{resumen}\n---")

        except Exception as e:
            LOG.warning(f"Recuperación falló en {etiqueta}: {e}")
    return "\n\n".join(bloques[:15])


## ======================
# PROMPT AVANZADO CON CITA Y REUTILIZACIÓN DE FUENTES
## ======================

PROMPT_ANALISIS = """
[Contexto recuperado con metadatos y referencias]
{contexto}

[Texto o consulta principal]
{texto}

[INSTRUCCIONES DE ANÁLISIS]
1️⃣ **Síntesis contextual**
   - Resume el caso y su contexto jurídico-médico.
   - Determina tipo de responsabilidad (civil, penal, laboral, bioética).
   - Reconoce las fuentes primarias y secundarias relevantes.

2️⃣ **Ejes conceptuales**
   - Desarrolla los temas: consentimiento informado, lex artis, causalidad, daño, prueba.
   - Para cada eje: explica el concepto, cita fuentes con autor/año/página o tribunal/año.
   - Usa las referencias documentales para sostener los razonamientos.

3️⃣ **Evaluación argumental**
   - Señala los argumentos más sólidos y las debilidades.
   - Indica contradicciones entre jurisprudencias o doctrinas.
   - Identifica lagunas normativas o probatorias.

4️⃣ **Fuentes y trazabilidad**
   - Crea una lista de fuentes relevantes con formato:
     “Autor – Título (Año), pág. X [URL o Jurisdicción]”
   - Indica si son doctrina, jurisprudencia o normativa.
   - Recomienda cuáles podrían ampliarse con consultas específicas o documentos adicionales.

5️⃣ **Preguntas derivadas**
   - Formula 3 a 5 preguntas que profundicen el análisis.
   - Indica qué información o tipo de documento sería necesario obtener para resolverlas.

6️⃣ **Conclusión jurídica**
   - Formula una tesis integrada, valorando la probabilidad de éxito (baja/media/alta).
   - Justifica con criterios jurídicos y médicos.

[FORMATO DE SALIDA]
Primero entrega un informe narrativo completo con secciones jerarquizadas:
🔹 Resumen contextual
🔹 Ejes conceptuales
🔹 Debilidades / Vacíos argumentales
🔹 Preguntas de profundización
🔹 Conclusión jurídica y fuentes

Luego, entrega un bloque JSON resumido de no más de 10 líneas con esta estructura:

{{
  "tesis": "...",
  "conceptos_clave": ["..."],
  "debilidades": ["..."],
  "preguntas": ["..."],
  "probabilidad_exito": "baja|media|alta",
  "fuentes_relevantes": [
    {{"autor": "...", "titulo": "...", "anio": "...", "pagina": "...", "url": "...", "tipo": "..."}}
  ]
}}
"""

## ======================
# INVOCACIÓN GEMINI CON EXTRACCIÓN ESTRUCTURADA DE FUENTES
## ======================

def _invoke_llm_json(llm: ChatGoogleGenerativeAI, system: str, user_prompt: str) -> Dict:
    """
    Ejecuta Gemini Pro en modo analítico-documental:
    - Produce narrativa + bloque JSON estructurado con citas.
    - Extrae referencias documentales reutilizables (fuentes_relevantes).
    """
    try:
        full_prompt = f"SYSTEM:\n{system}\n\nUSER:\n{user_prompt}"
        resp = llm.invoke(full_prompt)
        txt = resp.content if hasattr(resp, "content") else str(resp)

        # Detectar y extraer bloque JSON
        m = re.search(r'\{.*\}', txt, flags=re.S)
        parsed = {}
        if m:
            try:
                parsed = json.loads(m.group(0))
            except json.JSONDecodeError:
                parsed = {}

        razonamiento = txt.split("{")[0].strip()
        parsed["texto_completo"] = razonamiento or txt.strip()

        # Validación y formato mínimo
        parsed.setdefault("tesis", "(sin tesis detectada)")
        parsed.setdefault("conceptos_clave", [])
        parsed.setdefault("debilidades", [])
        parsed.setdefault("preguntas", [])
        parsed.setdefault("probabilidad_exito", "media")
        parsed.setdefault("fuentes_relevantes", [])

        # Normalizar fuentes_relevantes
        if isinstance(parsed["fuentes_relevantes"], list):
            fuentes_limpias = []
            for f in parsed["fuentes_relevantes"]:
                if isinstance(f, dict):
                    fuentes_limpias.append({
                        "autor": f.get("autor", ""),
                        "titulo": f.get("titulo", ""),
                        "anio": f.get("anio", ""),
                        "pagina": f.get("pagina", ""),
                        "url": f.get("url", ""),
                        "tipo": f.get("tipo", "")
                    })
                elif isinstance(f, str):
                    fuentes_limpias.append({"autor": f, "titulo": "", "anio": "", "pagina": "", "url": "", "tipo": ""})
            parsed["fuentes_relevantes"] = fuentes_limpias

        return parsed

    except Exception as e:
        return {
            "tesis": f"[LLM_ERROR] {e}",
            "conceptos_clave": [],
            "debilidades": [],
            "preguntas": [],
            "probabilidad_exito": "media",
            "fuentes_relevantes": [],
            "texto_completo": ""
        }

# ======================
# LLM WRAPPER (CREA EL MODELO GEMINI)
# ======================

from langchain_google_genai import ChatGoogleGenerativeAI

DEFAULT_LLM = "gemini-2.5-pro"

def _make_llm(model: str = DEFAULT_LLM, temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    """
    Inicializa el modelo Gemini con validación de clave y configuración de temperatura.
    """
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("⚠️ Falta GOOGLE_API_KEY en el entorno para usar Gemini.")
    return ChatGoogleGenerativeAI(model=model, temperature=temperature)

# ======================
# Núcleo de análisis (texto/PDF)
# ======================

def _build_context(query_or_text: str, retrievers: Dict[str, Chroma], k_por_base: int = 4) -> str:
    bloques = []
    for etiqueta, vs in retrievers.items():
        try:
            top = enhanced_similarity_search(vs, query_or_text, k=k_por_base, fetch_k=24)
            frag = "\n---".join([f"[{etiqueta}] {d.metadata.get('archivo_origen','?')} :: {d.page_content[:800]}" for d, _ in top])
            if frag:
                bloques.append(frag)
        except Exception as e:
            LOG.warning(f"Recuperación falló en {etiqueta}: {e}")
    return "\n\n".join(bloques[:8])

def analyse_text_medico(texto: str, retrievers: Dict[str, Chroma], llm: ChatGoogleGenerativeAI) -> Dict:
    contexto = _build_context(texto, retrievers)
    user_prompt = PROMPT_ANALISIS.format(contexto=contexto, texto=texto[:6000])
    return _invoke_llm_json(llm, SYSTEM_INSTRUCCIONES, user_prompt)

def analyse_pdf_medico(pdf_path: Path, retrievers: Dict[str, Chroma], llm: ChatGoogleGenerativeAI) -> Dict:
    if extract_text_from_pdf is None:
        raise RuntimeError("No se encontró extract_text_from_pdf (importa ingesta_mejorada_autonoma).")
    st = SettingsIngesta() if SettingsIngesta else None
    text = extract_text_from_pdf(pdf_path, st) if st else extract_text_from_pdf(pdf_path)
    if not text.strip():
        return {"tesis": "[ERROR] No se pudo extraer texto del PDF", "fundamentos": [], "riesgos": [], "prueba_faltante": [], "probabilidad_exito": "media", "notas": []}
    return analyse_text_medico(text, retrievers, llm)

# ======================
# Análisis avanzado iterativo (Nivel 2+)
# ======================

def analyse_deep_layer(result_json: dict, llm: ChatGoogleGenerativeAI, pregunta: str, nivel: int = 2) -> dict:
    """
    Nivel 2+: análisis jurídico profundo iterativo.
    Usa el JSON del resultado anterior como contexto y puede encadenarse recursivamente.
    """

    # Convertimos el resultado previo en texto para el modelo
    context = json.dumps(result_json, ensure_ascii=False, indent=2)

    # Construimos un prompt especializado
    prompt = f"""
    [Análisis previo]
    {context}

    [Nueva consulta]
    {pregunta}

    [Instrucciones]
    1) Usa el análisis previo como base factual. No repitas información.
    2) Profundiza el razonamiento jurídico: doctrina, jurisprudencia, interpretación normativa.
    3) Evalúa contradicciones, debilidades o mejoras posibles.
    4) Si procede, propone estrategias o líneas argumentales.
    5) Devuelve un JSON estructurado con:
       - "analisis_avanzado": texto razonado y extenso
       - "nuevos_fundamentos": lista de fundamentos adicionales
       - "nuevos_riesgos": lista de riesgos adicionales
       - "nivel": número de nivel actual
    """

    return _invoke_llm_json(
        llm,
        "Eres un jurista especializado en responsabilidad médica y análisis iterativo de jurisprudencia y doctrina.",
        prompt
    )

# ======================
# Generación de PDF (reporte)
# ======================

def _draw_wrapped_text(c: canvas.Canvas, text: str, x: float, y: float, max_width: float, leading: float = 14) -> float:
    from reportlab.pdfbase.pdfmetrics import stringWidth
    words = text.split()
    line = ""
    for w in words:
        test = f"{line} {w}".strip()
        if stringWidth(test, "Helvetica", 10) <= max_width:
            line = test
        else:
            c.drawString(x, y, line)
            y -= leading
            line = w
    if line:
        c.drawString(x, y, line)
        y -= leading
    return y

def render_pdf_report(output_path: Path, titulo: str, entrada: str, result: Dict, auditorias: List[VectorAudit]):
    c = canvas.Canvas(str(output_path), pagesize=A4)
    W, H = A4
    margin = 2*cm
    x = margin
    y = H - margin

    # Encabezado
    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, titulo)
    y -= 16
    c.setFont("Helvetica", 9)
    c.drawString(x, y, f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y -= 12
    c.drawString(x, y, "Analizador: Gemini 2.5 Pro + RAG sanitario multilingüe")
    y -= 18

    # Entrada
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Entrada (resumen):")
    y -= 14
    c.setFont("Helvetica", 10)
    y = _draw_wrapped_text(c, entrada[:1000], x, y, W - 2*margin)

    # Tesis
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Tesis (síntesis):")
    y -= 14
    c.setFont("Helvetica", 10)
    y = _draw_wrapped_text(c, result.get("tesis",""), x, y, W - 2*margin)

    # Fundamentos
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Fundamentos:")
    y -= 14
    c.setFont("Helvetica", 10)
    fundamentos = result.get("fundamentos", [])
    if not fundamentos: fundamentos = ["(sin fundamentos detectados)"]
    for f in fundamentos:
        if y < margin + 80:
            c.showPage(); y = H - margin
        y = _draw_wrapped_text(c, f"- {f}", x, y, W - 2*margin)

    # Riesgos
    c.setFont("Helvetica-Bold", 12)
    if y < margin + 60: c.showPage(); y = H - margin
    c.drawString(x, y, "Riesgos:")
    y -= 14
    c.setFont("Helvetica", 10)
    riesgos = result.get("riesgos", [])
    if not riesgos: riesgos = ["(no se identificaron riesgos relevantes)"]
    for r in riesgos:
        if y < margin + 60: c.showPage(); y = H - margin
        y = _draw_wrapped_text(c, f"- {r}", x, y, W - 2*margin)

    # Prueba faltante
    c.setFont("Helvetica-Bold", 12)
    if y < margin + 60: c.showPage(); y = H - margin
    c.drawString(x, y, "Prueba faltante:")
    y -= 14
    c.setFont("Helvetica", 10)
    faltante = result.get("prueba_faltante", [])
    if not faltante: faltante = ["(no se identificaron omisiones probatorias)"]
    for pf in faltante:
        if y < margin + 60: c.showPage(); y = H - margin
        y = _draw_wrapped_text(c, f"- {pf}", x, y, W - 2*margin)

    # Probabilidad de éxito
    c.setFont("Helvetica-Bold", 12)
    if y < margin + 40: c.showPage(); y = H - margin
    c.drawString(x, y, "Probabilidad de éxito:")
    y -= 14
    c.setFont("Helvetica", 10)
    c.drawString(x, y, result.get("probabilidad_exito", "media").upper())
    y -= 18

    # Auditoría vectorial
    c.setFont("Helvetica-Bold", 12)
    if y < margin + 80: c.showPage(); y = H - margin
    c.drawString(x, y, "Auditoría de bases vectoriales (resumen):")
    y -= 14
    c.setFont("Helvetica", 10)
    for a in auditorias:
        if y < margin + 70: c.showPage(); y = H - margin
        c.drawString(x, y, f"• {a.base}: frags={a.frags}, avg_words={a.avg_words:.1f}, "
                           f"diversidad={a.diversity:.2f}, cobertura={a.coverage_types:.2f}, rating={a.rating:.2f}")
        y -= 12

    # Texto completo del modelo (si existe)
    if "texto_completo" in result and result["texto_completo"].strip():
        if y < margin + 100:
            c.showPage(); y = H - margin
        c.setFont("Helvetica-Bold", 12)
        c.drawString(x, y, "Informe completo del modelo:")
        y -= 14
        c.setFont("Helvetica", 10)
        y = _draw_wrapped_text(c, result["texto_completo"][:4000], x, y, W - 2*margin)

    c.showPage()
    c.save()

# ======================
# CLI
# ======================

def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return path.read_text(errors="ignore")

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Analizador sanitario (Gemini Pro + RAG) con reporte PDF")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pdf", help="Ruta a PDF a analizar")
    g.add_argument("--text", help="Ruta a archivo de texto")
    g.add_argument("--query", help="Consulta breve a evaluar (sin archivo)")
    ap.add_argument("--tipo", choices=list(BASES_RAG.keys()), help="Base sanitaria prioritaria (opcional)")
    ap.add_argument("--out", required=True, help="Ruta de salida del PDF")
    ap.add_argument("--emb_model", default=DEFAULT_EMBEDDINGS, help="Modelo de embeddings HF")
    ap.add_argument("--llm", default=DEFAULT_LLM, help="Modelo Gemini (por defecto gemini-2.5-pro)")
    args = ap.parse_args()

    llm = _make_llm(model=args.llm)
    retrievers = _load_all_retrievers(DB_DIR_DEFAULT, BASES_RAG)
    if not retrievers:
        LOG.warning("No se encontraron bases sanitarias. Verifica 'chroma_db_legal/*'.")

    entrada = ""
    result = {}
    if args.pdf:
        entrada = f"[PDF] {args.pdf}"
        result = analyse_pdf_medico(Path(args.pdf), retrievers, llm)
    elif args.text:
        texto = _read_text_file(Path(args.text))
        entrada = f"[TEXT] {args.text}"
        result = analyse_text_medico(texto, retrievers, llm)
    elif args.query:
        entrada = f"[QUERY] {args.query}"
        result = analyse_text_medico(args.query, retrievers, llm)

    auditorias: List[VectorAudit] = []
    for nombre, vs in retrievers.items():
        auditorias.append(_audit_vectorstore(vs, nombre))

    out = Path(args.out)
    render_pdf_report(out, "Informe Sanitario Juridificado (Gemini + RAG)", entrada, result, auditorias)

    paquete = {
        "entrada": entrada,
        "resultado": result,
        "auditoria": [asdict(a) for a in auditorias],
        "timestamp": datetime.now().isoformat()
    }
    print(json.dumps(paquete, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
