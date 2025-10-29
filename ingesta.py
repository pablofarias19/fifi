# -*- coding: utf-8 -*-
"""
INGESTA MEJORADA CON VERSIONADO (jurídica)
-------------------------------------------
- OCR opcional (pytesseract + pdf2image)
- Extracción texto (pdfplumber / PyPDF2)
- Segmentación jurídica (considerandos, artículos, resolutivos, etc.)
- Validación + deduplicación por hash y score
- Almacenamiento en Chroma con versionado automático
- Integración completa con version_manager
"""

from __future__ import annotations
import re, hashlib, logging, json, uuid
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Imports de config centralizado
from config import (
    LOGS_DIR, IDIOMAS_VALIDOS, BASES_RAG,
    IngestaConfig, DEFAULT_INGESTA_CONFIG
)

# Sistema de versionado
from version_manager import (
    REGISTRY, BaseVersion, create_version_directory, get_version_directory
)
from embedding_validator import VALIDATOR

# ---------- Dependencias PDF / OCR ----------
try:
    import pdfplumber
    _HAS_PDFPLUMBER = True
except ImportError:
    _HAS_PDFPLUMBER = False

try:
    import PyPDF2
    _HAS_PYPDF2 = True
except ImportError:
    _HAS_PYPDF2 = False

try:
    from pdf2image import convert_from_path
    import pytesseract
    _HAS_OCR_STACK = True
except ImportError:
    _HAS_OCR_STACK = False

# ---------- LangChain / Chroma ----------
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# ======================
# Logging
# ======================

LOGS_DIR.mkdir(exist_ok=True)

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
# Tipos de Documentos
# ======================

class TipoDocumento(str, Enum):
    """Tipos de documentos jurídicos reconocidos"""
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
# Metadata Enriquecida
# ======================

@dataclass
class MetadataEnriquecido:
    """Metadata completa de documento jurídico"""
    archivo_origen: str
    hash_pdf: str
    id_doc: str = field(default_factory=lambda: str(uuid.uuid4()))
    fecha_ingestion: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Clasificación
    materia: str = ""
    fuente: str = ""
    tipo_documento: str = ""

    # Metadata jurídica
    tribunal: Optional[str] = None
    expediente: Optional[str] = None
    jurisdiccion: str = ""
    fecha_sentencia: Optional[str] = None
    fecha_resolucion: Optional[str] = None

    # Metadata documental
    autor: str = ""
    idioma: str = "es"
    resumen: str = ""

    # Estadísticas
    tokens: int = 0
    fragmentos_extraidos: int = 0
    fragmentos_validos: int = 0
    score_promedio: float = 0.0

    # Observaciones
    observaciones: str = ""

    def to_dict(self) -> dict:
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
            "fecha_resolucion": self.fecha_resolucion,
            "fragmentos_extraidos": self.fragmentos_extraidos,
            "fragmentos_validos": self.fragmentos_validos,
            "score_promedio": self.score_promedio
        }

# ======================
# Extracción de texto
# ======================

def _extract_pdfplumber(pdf_path: Path) -> str:
    """Extrae texto usando pdfplumber"""
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


def _extract_pypdf2(pdf_path: Path) -> str:
    """Extrae texto usando PyPDF2"""
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


def _extract_ocr(pdf_path: Path, config: IngestaConfig) -> Tuple[str, float]:
    """Extrae texto con OCR"""
    if not _HAS_OCR_STACK:
        return "", 0.0
    try:
        images = convert_from_path(str(pdf_path), dpi=config.dpi_ocr)
        chunks, quals = [], []
        for img in images:
            txt = pytesseract.image_to_string(img, lang=config.lang_ocr)
            chunks.append(txt)
            qual = min(1.0, len(txt) / 1000.0) if txt.strip() else 0.0
            quals.append(qual)
        full_text = "\n".join(chunks)
        avg_quality = sum(quals) / len(quals) if quals else 0.0
        return full_text, avg_quality
    except Exception as e:
        log.warning(f"OCR falló en {pdf_path.name}: {e}")
        return "", 0.0


def extract_text_from_pdf(pdf_path: Path, config: IngestaConfig = DEFAULT_INGESTA_CONFIG) -> str:
    """Extrae texto combinando métodos disponibles"""
    text = ""

    # 1. Intentar pdfplumber primero
    text = _extract_pdfplumber(pdf_path)

    # 2. Si falla, intentar PyPDF2
    if not text.strip():
        text = _extract_pypdf2(pdf_path)

    # 3. Si aún falla y OCR está habilitado, usar OCR
    if not text.strip() and config.usar_ocr:
        ocr_text, _ = _extract_ocr(pdf_path, config)
        text = ocr_text

    return text

# ======================
# Validación y clasificación
# ======================

def validate_pdf(pdf_path: Path) -> bool:
    """Valida que el PDF sea válido"""
    try:
        if not pdf_path.exists():
            return False
        if pdf_path.stat().st_size < 1024:
            return False
        with open(pdf_path, "rb") as f:
            head = f.read(4)
        return head == b"%PDF"
    except Exception:
        return False


def classify_document(text: str) -> str:
    """Clasifica tipo de documento jurídico"""
    low = text.lower()
    if re.search(_PATTERNS["sentencia"], low):
        return TipoDocumento.SENTENCIA.value
    if "decreto" in low:
        return TipoDocumento.DECRETO.value
    if "resolución" in low or "resolucion" in low:
        return TipoDocumento.RESOLUCION.value
    if re.search(_PATTERNS["expediente"], low):
        return TipoDocumento.EXPEDIENTE.value
    return TipoDocumento.DESCONOCIDO.value


def detect_metadata_enriched(pdf_path: Path, text: str) -> MetadataEnriquecido:
    """Extrae metadatos del PDF y texto"""
    # Hash del PDF
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

    # Tribunal
    mtrib = re.search(_PATTERNS["tribunal"], text)
    meta.tribunal = mtrib.group(0) if mtrib else None

    return meta

# ======================
# Segmentación jurídica
# ======================

def _estimate_quality(text: str) -> float:
    """Score de calidad de fragmento"""
    if not text:
        return 0.0
    n = len(text.split())
    score = 0.3 if n >= 20 else 0.0
    if 80 <= n <= 1200:
        score += 0.3
    # Señales jurídicas
    hits = sum(1 for k in ("considerando", "artículo", "articulo", "resuelve", "sentencia") if k in text.lower())
    score += min(0.4, hits * 0.1)
    return max(0.0, min(1.0, score))


def _segment_legal(text: str, config: IngestaConfig) -> List[Document]:
    """Segmentación jurídica con preservación de estructura"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". "]
    )

    # Detectar tipo de línea
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    buckets: List[Tuple[str, List[str]]] = []
    current_type, buf = "parrafo", []

    def flush():
        nonlocal buf, current_type, buckets
        if buf:
            buckets.append((current_type, buf))
            buf = []

    for ln in lines:
        ntype = ("considerando" if re.match(_PATTERNS["considerando"], ln)
                 else "articulo" if re.match(_PATTERNS["articulo"], ln)
                 else "resolutivo" if re.match(_PATTERNS["resolutivo"], ln)
                 else "parrafo")
        if ntype != current_type:
            flush()
            current_type = ntype
        buf.append(ln)
    flush()

    # Chunking por tipo
    docs: List[Document] = []
    for t, lines_block in buckets:
        big_block = "\n".join(lines_block)
        for chunk in splitter.split_text(big_block):
            q = _estimate_quality(chunk)
            if q < config.min_quality_score:
                continue
            docs.append(Document(
                page_content=chunk,
                metadata={"tipo_estructura": t, "score_calidad": q}
            ))

    return docs


def _dedup_by_hash(docs: List[Document]) -> List[Document]:
    """Deduplica fragmentos por hash MD5"""
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
# Pipeline principal con versionado
# ======================

def ingest_pdf_to_chroma(
    pdf_path: Path,
    base_nombre: str,
    materia: Optional[str] = None,
    config: IngestaConfig = DEFAULT_INGESTA_CONFIG
) -> Dict[str, Any]:
    """
    Ingesta PDF con versionado automático.

    Args:
        pdf_path: Ruta al PDF
        base_nombre: Nombre de la base (ej: "Salud - Médica")
        materia: Materia/rama del derecho
        config: Configuración de ingesta

    Returns:
        Diccionario con resultado
    """
    assert base_nombre in BASES_RAG, f"Base desconocida: {base_nombre}"

    # 1. Validar PDF
    if not validate_pdf(pdf_path):
        return {"ok": False, "error": "PDF inválido o dañado"}

    # 2. Extraer texto
    text = extract_text_from_pdf(pdf_path, config)
    if not text.strip():
        return {"ok": False, "error": "Sin texto utilizable"}

    # 3. Segmentar
    docs = _segment_legal(text, config)
    if not docs:
        return {"ok": False, "error": "Segmentación sin fragmentos válidos"}

    # 4. Deduplicar
    docs = _dedup_by_hash(docs)

    # 5. Metadata enriquecida
    meta_doc = detect_metadata_enriched(pdf_path, text)
    meta_doc.materia = materia or base_nombre
    meta_doc.fragmentos_extraidos = len(docs)

    if docs:
        meta_doc.score_promedio = sum(d.metadata.get("score_calidad", 0.5) for d in docs) / len(docs)
        meta_doc.fragmentos_validos = len(docs)

    # 6. Asignar metadata a cada fragmento
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

    # 7. Gestionar versión
    active_version = REGISTRY.get_active_version(base_nombre)

    if not active_version:
        # Primera versión de esta base
        active_version = "1.0.0"
        version_dir = create_version_directory(base_nombre, active_version)

        # Registrar versión inicial
        version_meta = BaseVersion(
            version=active_version,
            embedding_model=config.embeddings_model if hasattr(config, 'embeddings_model') else VALIDATOR._embedding_cache.keys().__iter__().__next__() if VALIDATOR._embedding_cache else "sentence-transformers/all-mpnet-base-v2",
            embedding_dim=768,
            total_docs=1,
            total_fragments=len(docs),
            created_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
            migration_from=None,
            quality_score=meta_doc.score_promedio,
            is_active=True
        )
        REGISTRY.register_version(base_nombre, version_meta)
    else:
        version_dir = get_version_directory(base_nombre, active_version)

    # 8. Cargar/crear vectorstore
    embeddings = VALIDATOR.get_embeddings()
    vs = Chroma(
        collection_name="legal_fragments",
        embedding_function=embeddings,
        persist_directory=str(version_dir)
    )

    # 9. Insertar documentos
    vs.add_documents(docs)

    log.info(f"✅ Ingesta exitosa: {pdf_path.name} → {base_nombre} v{active_version} ({len(docs)} fragmentos)")

    return {
        "ok": True,
        "base": base_nombre,
        "version": active_version,
        "n_fragmentos": len(docs),
        "metadata": meta_doc.to_dict()
    }

# ======================
# CLI
# ======================

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Ingesta mejorada con versionado")
    ap.add_argument("--pdf", required=True, help="Ruta al PDF")
    ap.add_argument("--base", required=True, choices=list(BASES_RAG.keys()))
    ap.add_argument("--materia", default=None)
    ap.add_argument("--no-ocr", action="store_true", help="Desactiva OCR")
    args = ap.parse_args()

    config = DEFAULT_INGESTA_CONFIG
    if args.no_ocr:
        config.usar_ocr = False

    res = ingest_pdf_to_chroma(Path(args.pdf), args.base, args.materia, config)
    print(json.dumps(res, ensure_ascii=False, indent=2))
