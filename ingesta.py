# -*- coding: utf-8 -*-
"""
INGESTA MEJORADA AUTÓNOMA (jurídica)
------------------------------------
- OCR opcional (pytesseract + pdf2image)
- Extracción texto (pdfplumber / PyPDF2)
- Segmentación jurídica (considerandos, artículos, resolutivos, etc.)
- Validación + deduplicación por hash y score
- Almacenamiento en Chroma (vectorstore)
- Settings centralizada
- Funciones clave exportadas:
    extract_text_from_pdf()
    validate_pdf()
    classify_document()
    detect_metadata_enriched()
- Lista de bases RAG por materia (BASES_RAG)
- Modular y compatible con: LangChain, Chroma, Streamlit y analyser.py
"""

from __future__ import annotations
import os, re, io, gc, json, hashlib, logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import uuid
from enum import Enum

# ---------- Dependencias PDF / OCR ----------
try:
    import pdfplumber
    _HAS_PDFPLUMBER = True
except Exception:
    _HAS_PDFPLUMBER = False

try:
    import PyPDF2
    _HAS_PYPDF2 = True
except Exception:
    _HAS_PYPDF2 = False

try:
    from pdf2image import convert_from_path
    import pytesseract
    _HAS_OCR_STACK = True
except Exception:
    _HAS_OCR_STACK = False

# ---------- LangChain / Chroma ----------
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings as SentenceTransformerEmbeddings
from langchain_chroma import Chroma


# ======================
# Rutas y logging
# ======================

ROOT = Path(__file__).parent
PDF_DIR_DEFAULT = ROOT / "pdfs"
DB_DIR_DEFAULT = ROOT / "chroma_db_legal"
CACHE_DIR_DEFAULT = ROOT / "chroma_cache"
LOGS_DIR = ROOT / "logs"
for d in [PDF_DIR_DEFAULT, DB_DIR_DEFAULT, CACHE_DIR_DEFAULT, LOGS_DIR]:
    d.mkdir(exist_ok=True)

def make_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        fh = logging.FileHandler(LOGS_DIR / f"{name}.log", encoding="utf-8")
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

log = make_logger("ingesta")

# ======================
# Idiomas y configuración global
# ======================

IDIOMAS_VALIDOS = ["en", "de", "es", "fr", "it"]
LANG_TESSERACT = "spa+eng+fra+ita+deu"  # OCR multilingüe

# ======================
# Bases RAG disponibles
# ======================

BASES_RAG: Dict[str, str] = {
    "Salud - Médica": "pdfs_salud_medica",
    "Salud - Laboral": "pdfs_salud_laboral",
    "Jurisprudencia - Salud": "pdfs_jurisprud_salud",
    "Legislación - Salud": "pdfs_ley_salud"
}


# ======================
# Settings y tipos
# ======================

class TipoDocumento(str, Enum):
    SENTENCIA = "sentencia"
    FALLO = "fallo"
    DECRETO = "decreto"
    RESOLUCION = "resolucion"
    AUTO = "auto"
    PROVIDENCIA = "providencia"
    EXPEDIENTE = "expediente"
    CONTRATO = "contrato"
    DEMANDA = "demanda"
    ESCRITO = "escrito"
    DICTAMEN = "dictamen"
    DESCONOCIDO = "desconocido"
# ======================
# Settings y tipos
# ======================

class TipoDocumento(str, Enum):
    SENTENCIA = "sentencia"
    FALLO = "fallo"
    DECRETO = "decreto"
    RESOLUCION = "resolucion"
    AUTO = "auto"
    PROVIDENCIA = "providencia"
    EXPEDIENTE = "expediente"
    CONTRATO = "contrato"
    DEMANDA = "demanda"
    ESCRITO = "escrito"
    DICTAMEN = "dictamen"
    DESCONOCIDO = "desconocido"


@dataclass
class Settings:
    # Rutas
    pdf_dir: Path = field(default_factory=lambda: PDF_DIR_DEFAULT)
    db_dir: Path = field(default_factory=lambda: DB_DIR_DEFAULT)
    cache_dir: Path = field(default_factory=lambda: CACHE_DIR_DEFAULT)
    # Chunking
    chunk_size: int = 600
    chunk_overlap: int = 120
    # OCR
    usar_ocr: bool = True
    dpi_ocr: int = 300
    lang_ocr: str = LANG_TESSERACT
    # Embeddings / RAG
    embeddings_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    search_k: int = 8
    # Calidad mínima
    min_quality_score: float = 0.30

    def ensure_dirs(self) -> None:
        self.pdf_dir.mkdir(exist_ok=True)
        self.db_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
    calidad_ocr: float = 1.0
    calidad_texto: float = 0.5
    fragmentos_extraidos: int = 0
    fragmentos_validos: int = 0
    score_promedio: float = 0.0
    metodo_analisis: str = "ingesta_mejorada_autonoma"
    fecha_ingestion: str = field(default_factory=lambda: datetime.now().isoformat())


# ======================
# Patrones jurídicos
# ======================

_PATTERNS = {
    "considerando": re.compile(r"(?i)^\s*(visto\s+y\s+)?considerando[s]?\b.*", re.MULTILINE),
    "articulo": re.compile(r"(?i)^\s*(art[íi]culo[s]?|art\.)\s+\d+\b.*", re.MULTILINE),
    "resolutivo": re.compile(r"(?i)^\s*(se\s+)?resuelve\b.*", re.MULTILINE),
    "expediente": re.compile(r"(?i)(expediente|expte|exp\.)\s*[nº°#:]?\s*([\w\-/\.]+)"),
    "fecha": re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{1,2}\s+de\s+\w+\s+de\s+\d{4}\b", re.IGNORECASE),
    "tribunal": re.compile(r"(?i)\b(juzgado|c[aá]mara|tribunal|corte)\b.*?(civil|laboral|comercial|federal)"),
    "sentencia": re.compile(r"(?i)\b(sentencia|fallo)\b"),
}


# ======================
# Extracción de texto
# ======================

def _extract_pdfplumber(pdf_path: Path) -> str:
    if not _HAS_PDFPLUMBER:
        return ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        log.warning(f"pdfplumber falló en {pdf_path.name}: {e}")
        return ""


def _extract_ocr(pdf_path: Path, dpi: int = 300, lang: str = LANG_TESSERACT) -> Tuple[str, float]:
    if not _HAS_OCR_STACK:
        return "", 0.0
    try:
        images = convert_from_path(str(pdf_path), dpi=dpi)
        chunks, quals = [], []
        for img in images:
            txt = pytesseract.image_to_string(img, lang=lang)
            chunks.append(txt)
            # Estimate quality based on text length and character diversity
            qual = min(1.0, len(txt) / 1000.0) if txt.strip() else 0.0
            quals.append(qual)
        full_text = "\n".join(chunks)
        avg_quality = sum(quals) / len(quals) if quals else 0.0
        return full_text, avg_quality
    except Exception as e:
        log.warning(f"OCR falló en {pdf_path.name}: {e}")
        return "", 0.0


def extract_text_from_pdf(pdf_path: Path, settings: Settings) -> str:
    """Extrae texto combinando métodos: pdfplumber, PyPDF2, OCR opcional."""
    text = ""
    
    # Try pdfplumber first
    text = _extract_pdfplumber(pdf_path)
    
    # If no text, try PyPDF2
    if not text.strip():
        text = _extract_pypdf2(pdf_path)
    
    # If still no text and OCR is enabled, try OCR
    if not text.strip() and settings.usar_ocr:
        ocr_text, _ = _extract_ocr(pdf_path, settings.dpi_ocr, settings.lang_ocr)
        text = ocr_text
    
    return text


@dataclass
class MetadataEnriquecido:
    """
    Estructura de metadatos enriquecidos para cada documento jurídico.
    Incluye trazabilidad, idioma, fuente, tipo y fecha de ingreso.
    """

    archivo_origen: str                # Nombre del archivo PDF
    hash_pdf: str                      # Hash MD5 único del PDF
    materia: str = ""                  # Rama del derecho o base RAG destino
    fuente: str = ""                   # Origen del documento (jurisprudencia, doctrina, etc.)
    idioma: str = "es"                 # Idioma detectado o definido
    tipo_documento: str = ""           # Tipo de documento (fallo, artículo, ley, etc.)
    resumen: str = ""                  # Resumen breve generado automáticamente o manualmente
    tokens: int = 0                    # Cantidad de tokens extraídos o procesados

    # 🔒 Nuevos campos de trazabilidad
    id_doc: str = field(default_factory=lambda: str(uuid.uuid4()))
    fecha_ingestion: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # 🧠 Campos opcionales futuros
    autor: str = ""
    jurisdiccion: str = ""
    tribunal: str = ""
    expediente: str = ""
    observaciones: str = ""
    fecha_sentencia: str = ""
    fecha_resolucion: str = ""

    def to_dict(self):
        return {
            "id_doc": self.id_doc,
            "archivo_origen": self.archivo_origen,
            "hash_pdf": self.hash_pdf,
            "materia": self.materia,
            "fuente": self.fuente,
            "idioma": self.idioma,
            "tipo_documento": self.tipo_documento,
            "resumen": self.resumen,
            "tokens": self.tokens,
            "autor": self.autor,
            "jurisdiccion": self.jurisdiccion,
            "tribunal": self.tribunal,
            "expediente": self.expediente,
            "observaciones": self.observaciones,
            "fecha_ingestion": self.fecha_ingestion,
            "fecha_sentencia": self.fecha_sentencia,
            "fecha_resolucion": self.fecha_resolucion
        }


def _extract_pypdf2(pdf_path: Path) -> str:
    if not _HAS_PYPDF2:
        return ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        log.warning(f"PyPDF2 falló en {pdf_path.name}: {e}")
        return ""





# ======================
# Validación y clasificación
# ======================

def validate_pdf(pdf_path: Path) -> bool:
    """Valida existencia, tamaño y encabezado básico."""
    try:
        if not pdf_path.exists(): return False
        if pdf_path.stat().st_size < 1024: return False
        with open(pdf_path, "rb") as f:
            head = f.read(4)
        return head == b"%PDF"
    except Exception:
        return False


def classify_document(text: str) -> str:
    """Clasifica tipo jurídico básico por reglas rápidas."""
    low = text.lower()
    if re.search(_PATTERNS["sentencia"], low): return TipoDocumento.SENTENCIA.value
    if "decreto" in low: return TipoDocumento.DECRETO.value
    if "resolución" in low or "resolucion" in low: return TipoDocumento.RESOLUCION.value
    if re.search(_PATTERNS["expediente"], low): return TipoDocumento.EXPEDIENTE.value
    return TipoDocumento.DESCONOCIDO.value


def _estimate_quality(text: str) -> float:
    """Score simple de calidad de fragmento: tamaño + señales jurídicas."""
    if not text: return 0.0
    n = len(text.split())
    score = 0.3 if n >= 20 else 0.0
    if 80 <= n <= 1200: score += 0.3
    # señales
    hits = sum(1 for k in ("considerando", "artículo", "articulo", "resuelve", "sentencia") if k in text.lower())
    score += min(0.4, hits * 0.1)
    return max(0.0, min(1.0, score))


def _segment_legal(text: str, settings: Settings) -> List[Document]:
    """Segmentación jurídica avanzada preservando tipos (considerando, artículo, resolutivo, párrafo)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". "]
    )
    # Marca de tipo aproximada por líneas pivote
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    buckets: List[Tuple[str, List[str]]] = []
    current_type, buf = "parrafo", []
    def flush():
        nonlocal buf, current_type, buckets
        if buf:
            buckets.append((current_type, buf))
            buf = []

    for ln in lines:
        lt = ln.lower()
        ntype = ("considerando" if re.match(_PATTERNS["considerando"], ln)
                 else "articulo" if re.match(_PATTERNS["articulo"], ln)
                 else "resolutivo" if re.match(_PATTERNS["resolutivo"], ln)
                 else "parrafo")
        if ntype != current_type:
            flush()
            current_type = ntype
        buf.append(ln)
    flush()

    docs: List[Document] = []
    for t, lines_block in buckets:
        big_block = "\n".join(lines_block)
        for chunk in splitter.split_text(big_block):
            q = _estimate_quality(chunk)
            if q < settings.min_quality_score:
                continue
            docs.append(Document(page_content=chunk, metadata={"tipo_estructura": t, "score_calidad": q}))
    return docs


def detect_metadata_enriched(pdf_path: Path, text: str) -> MetadataEnriquecido:
    """Extrae metadatos clave (expediente, tribunal, fechas, tipo_documento, etc.)."""
    # Hash
    with open(pdf_path, "rb") as f:
        h = hashlib.md5(f.read()).hexdigest()

    meta = MetadataEnriquecido(archivo_origen=pdf_path.name, hash_pdf=h)
    meta.tipo_documento = classify_document(text)
    # Expediente
    mexp = re.search(_PATTERNS["expediente"], text)
    meta.expediente = mexp.group(2) if mexp else None
    # Fechas
    mfecha = re.search(_PATTERNS["fecha"], text)
    if mfecha:
        fstr = mfecha.group(0)
        if meta.tipo_documento == TipoDocumento.SENTENCIA.value:
            meta.fecha_sentencia = fstr
        else:
            meta.fecha_resolucion = fstr
    # Tribunal aprox
    mtrib = re.search(_PATTERNS["tribunal"], text)
    meta.tribunal = mtrib.group(0) if mtrib else None
    return meta


# ======================
# Vectorstore (Chroma)
# ======================

def _ensure_vectorstore(db_dir: Path, embeddings_model: str) -> Chroma:

    # ============================================================
    # Crear el vectorstore con detección de dimensión segura
    # ============================================================
    embeddings = SentenceTransformerEmbeddings(model_name=embeddings_model)

    # Detección de dimensión (compatible con versiones viejas y nuevas)
    try:
        if hasattr(embeddings, "_client"):
            model_ref = embeddings._client
        elif hasattr(embeddings, "client"):
            model_ref = embeddings.client
        else:
            model_ref = None

        if model_ref and hasattr(model_ref, "get_sentence_embedding_dimension"):
            dim = model_ref.get_sentence_embedding_dimension()
        elif model_ref and hasattr(model_ref, "encode"):
            # Método alternativo: generar embedding de prueba
            test_vec = model_ref.encode("dim_test")
            dim = len(test_vec)
        else:
            dim = "desconocida"
    except Exception:
        dim = "desconocida"

    print(f"[Chroma] Creando vectorstore en {db_dir} con modelo: {embeddings_model} (dim={dim})")

    vs = Chroma(
        collection_name="legal_fragments",
        embedding_function=embeddings,
        persist_directory=str(db_dir)
    )
    return vs


def _dedup_by_hash(docs: List[Document]) -> List[Document]:
    seen = set()
    out: List[Document] = []
    for d in docs:
        h = hashlib.md5(d.page_content.encode("utf-8")).hexdigest()
        if h in seen: 
            continue
        seen.add(h)
        d.metadata["hash_fragmento"] = h
        out.append(d)
    return out


# ======================
# Pipeline principal
# ======================

def ingest_pdf_to_chroma(
    pdf_path: Path,
    base_nombre: str,
    materia: Optional[str],
    settings: Optional[Settings] = None
) -> Dict[str, Any]:
    """
    Ingiera un PDF:
      - Valida PDF
      - Extrae texto (con OCR opcional)
      - Segmenta jurídicamente
      - Valida + dedup
      - Inserta en Chroma con metadatos enriquecidos
    """
    settings = settings or Settings()
    settings.ensure_dirs()
    assert base_nombre in BASES_RAG, f"Base desconocida: {base_nombre}"

    if not validate_pdf(pdf_path):
        return {"ok": False, "error": "PDF inválido o dañado"}

    text = extract_text_from_pdf(pdf_path, settings)
    if not text.strip():
        return {"ok": False, "error": "Sin texto utilizable (ni OCR)"}

    # Segmentación legal
    docs = _segment_legal(text, settings)

    if not docs:
        return {"ok": False, "error": "Segmentación sin fragmentos válidos"}

    docs = _dedup_by_hash(docs)

    # Metadatos enriquecidos (a nivel doc)
    meta_doc = detect_metadata_enriched(pdf_path, text)
    meta_doc.materia = materia
    meta_doc.fragmentos_extraidos = len(docs)

    # Score promedio simple
    if docs:
        meta_doc.score_promedio = sum(d.metadata.get("score_calidad", 0.5) for d in docs) / len(docs)
        meta_doc.fragmentos_validos = len(docs)

    # Asignar metadatos globales a cada fragmento
    for d in docs:
        d.metadata.update({
            "archivo_origen": meta_doc.archivo_origen,
            "hash_pdf": meta_doc.hash_pdf,
            "tipo_documento": meta_doc.tipo_documento,
            "materia": meta_doc.materia,
            "tribunal": meta_doc.tribunal,
            "expediente": meta_doc.expediente,
            "fecha_sentencia": meta_doc.fecha_sentencia,
            "fecha_resolucion": meta_doc.fecha_resolucion,
            "fecha_ingestion": meta_doc.fecha_ingestion,
            "base_nombre": base_nombre
        })
    # Persistir en Chroma
    vs = _ensure_vectorstore(settings.db_dir / BASES_RAG[base_nombre], settings.embeddings_model)
    vs.add_documents(docs)  # En Chroma 0.4+ persiste solo
    return {"ok": True, "base": base_nombre, "n_fragmentos": len(docs), "metadata": meta_doc.to_dict()}


# ======================
# CLI (opcional)
# ======================

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Ingesta mejorada autónoma (jurídica)")
    ap.add_argument("--pdf", required=True, help="Ruta al PDF")
    ap.add_argument("--base", required=True, choices=list(BASES_RAG.keys()))
    ap.add_argument("--materia", default=None)
    ap.add_argument("--no-ocr", action="store_true", help="Desactiva OCR")
    ap.add_argument("--model", default=None, help="Modelo embeddings HuggingFace")
    args = ap.parse_args()

    st = Settings(usar_ocr=not args.no_ocr)
    if args.model: st.embeddings_model = args.model
    res = ingest_pdf_to_chroma(Path(args.pdf), args.base, args.materia, st)
    print(json.dumps(res, ensure_ascii=False, indent=2))
