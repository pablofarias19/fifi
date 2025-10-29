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

    # Metadata documental (⭐ NUEVOS CAMPOS)
    autor: str = ""
    titulo: str = ""              # ⭐ Título del documento
    anio: Optional[str] = None    # ⭐ Año del documento
    idioma: str = "es"
    resumen: str = ""
    url: str = ""                 # ⭐ URL si está disponible
    num_paginas: int = 0          # ⭐ Número de páginas del PDF

    # Estadísticas
    tokens: int = 0
    fragmentos_extraidos: int = 0
    fragmentos_validos: int = 0
    score_promedio: float = 0.0

    # Observaciones
    observaciones: str = ""
    metodo_deteccion: str = "regex"  # ⭐ "regex" o "llm" para trazabilidad

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
            "titulo": self.titulo,
            "anio": self.anio,
            "url": self.url,
            "num_paginas": self.num_paginas,
            "jurisdiccion": self.jurisdiccion,
            "tribunal": self.tribunal,
            "expediente": self.expediente,
            "observaciones": self.observaciones,
            "fecha_ingestion": self.fecha_ingestion,
            "fecha_sentencia": self.fecha_sentencia,
            "fecha_resolucion": self.fecha_resolucion,
            "fragmentos_extraidos": self.fragmentos_extraidos,
            "fragmentos_validos": self.fragmentos_validos,
            "score_promedio": self.score_promedio,
            "metodo_deteccion": self.metodo_deteccion
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


# ======================
# Detección Avanzada de Metadata
# ======================

def detect_author_advanced(text: str) -> Optional[str]:
    """
    Detecta autor con patrones mejorados.

    Patrones soportados:
    - "Autor: Dr. Juan Pérez"
    - "Autores: María García y Juan López"
    - "Dr./Dra./Prof. Nombre Apellido"
    - "Por: Nombre Apellido"
    """
    # Limpiar y tomar primeras 2000 caracteres (el autor suele estar al inicio)
    sample = text[:2000]

    patterns = [
        r"(?:Autor(?:es)?|Por):\s*(?:Dr\.|Dra\.|Prof\.)?\s*([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){1,3})",
        r"(?:Dr\.|Dra\.|Prof\.)\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)",
        r"(?:Firmado|Suscrito)\s+por:?\s*([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){1,3})",
    ]

    for pattern in patterns:
        match = re.search(pattern, sample, re.IGNORECASE)
        if match:
            autor = match.group(1).strip()
            # Validar que no sea muy corto ni muy largo
            if 5 <= len(autor) <= 50:
                return autor

    return None


def detect_title_advanced(text: str) -> Optional[str]:
    """
    Detecta título del documento.

    Estrategias:
    1. Primera línea en mayúsculas (mín 10 chars)
    2. Línea entre "TÍTULO:" o similar
    3. Primera línea con formato título (Title Case)
    """
    lines = [l.strip() for l in text[:1500].split("\n") if l.strip()]

    if not lines:
        return None

    # Estrategia 1: Buscar patrón "TÍTULO:" o similar
    for i, line in enumerate(lines[:10]):
        if re.match(r"(?i)^(?:t[íi]tulo|asunto|materia):\s*(.+)", line):
            titulo = re.match(r"(?i)^(?:t[íi]tulo|asunto|materia):\s*(.+)", line).group(1)
            if 10 <= len(titulo) <= 200:
                return titulo

    # Estrategia 2: Primera línea en mayúsculas
    for line in lines[:5]:
        if line.isupper() and 10 <= len(line) <= 200:
            return line

    # Estrategia 3: Primera línea con formato de título (inicia con mayúscula)
    for line in lines[:3]:
        if line[0].isupper() and 10 <= len(line) <= 200:
            # Verificar que no sea una frase común
            if not any(skip in line.lower() for skip in ["visto", "autos", "considerando", "resultando"]):
                return line

    return None


def detect_year_advanced(text: str) -> Optional[str]:
    """
    Detecta año del documento.

    Busca años entre 1900-2099, priorizando:
    1. Años en contexto de fecha formal
    2. Año más frecuente en el documento
    """
    # Buscar todos los años (1900-2099)
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", text)

    if not years:
        return None

    # Buscar año en contexto de fecha formal (más confiable)
    formal_date_patterns = [
        r"(?:de|del año|año)\s+(19\d{2}|20\d{2})",
        r"(19\d{2}|20\d{2})\s*[-/]\s*\d{1,2}\s*[-/]\s*\d{1,2}",
        r"\d{1,2}\s*[-/]\s*\d{1,2}\s*[-/]\s*(19\d{2}|20\d{2})",
    ]

    for pattern in formal_date_patterns:
        matches = re.findall(pattern, text[:2000])
        if matches:
            # Devolver el año más reciente en contexto formal
            return max(matches)

    # Si no hay contexto formal, devolver año más frecuente
    from collections import Counter
    year_counts = Counter(years)
    most_common_year = year_counts.most_common(1)[0][0]

    return most_common_year


def detect_jurisdiction_advanced(text: str) -> Optional[str]:
    """
    Detecta jurisdicción (provincia/país).

    Busca:
    - Provincias argentinas
    - Ciudades principales
    - País si es internacional
    """
    sample = text[:3000]

    # Provincias argentinas
    provincias = [
        "Buenos Aires", "CABA", "Ciudad Autónoma de Buenos Aires",
        "Córdoba", "Santa Fe", "Mendoza", "Tucumán", "Entre Ríos",
        "Salta", "Chaco", "Corrientes", "Misiones", "San Juan",
        "Jujuy", "Río Negro", "Neuquén", "Formosa", "Chubut",
        "San Luis", "Catamarca", "La Rioja", "La Pampa",
        "Santa Cruz", "Tierra del Fuego"
    ]

    # Buscar provincia en contexto
    for provincia in provincias:
        patterns = [
            rf"\b{re.escape(provincia)}\b",
            rf"Provincia\s+de\s+{re.escape(provincia)}",
            rf"Juzgado.*{re.escape(provincia)}",
            rf"Tribunal.*{re.escape(provincia)}",
        ]

        for pattern in patterns:
            if re.search(pattern, sample, re.IGNORECASE):
                return provincia

    # Buscar "Argentina" si no hay provincia
    if re.search(r"\bArgentina\b", sample):
        return "Argentina"

    return None


def detect_language(text: str) -> str:
    """
    Detecta idioma del texto usando langdetect.
    Fallback a "es" si falla la detección.
    """
    try:
        # Intentar importar langdetect
        from langdetect import detect, LangDetectException

        # Muestrear hasta 2000 caracteres
        sample = text[:2000]

        if not sample.strip():
            return "es"

        lang = detect(sample)

        # Mapear códigos a los soportados
        if lang in IDIOMAS_VALIDOS:
            return lang

        # Si no está en lista válida, devolver español por defecto
        return "es"

    except (ImportError, LangDetectException, Exception):
        # Si langdetect no está instalado o falla, devolver español
        return "es"


def extract_pdf_metadata(pdf_path: Path) -> dict:
    """
    Extrae metadata del PDF (número de páginas, metadata del archivo).

    Returns:
        dict con: num_paginas, autor_pdf, titulo_pdf, etc.
    """
    metadata = {
        "num_paginas": 0,
        "autor_pdf": None,
        "titulo_pdf": None,
    }

    try:
        if _HAS_PDFPLUMBER:
            with pdfplumber.open(pdf_path) as pdf:
                metadata["num_paginas"] = len(pdf.pages)

                # Intentar extraer metadata del PDF
                if hasattr(pdf, "metadata") and pdf.metadata:
                    metadata["autor_pdf"] = pdf.metadata.get("Author", None)
                    metadata["titulo_pdf"] = pdf.metadata.get("Title", None)

        elif _HAS_PYPDF2:
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                metadata["num_paginas"] = len(reader.pages)

                if reader.metadata:
                    metadata["autor_pdf"] = reader.metadata.get("/Author", None)
                    metadata["titulo_pdf"] = reader.metadata.get("/Title", None)

    except Exception as e:
        log.warning(f"No se pudo extraer metadata del PDF: {e}")

    return metadata


def extract_metadata_with_llm(text: str, pdf_path: Path, config: IngestaConfig) -> dict:
    """
    Extrae metadatos usando Gemini Pro (opcional, lento pero preciso).

    Args:
        text: Texto del documento
        pdf_path: Ruta al PDF
        config: Configuración de ingesta

    Returns:
        dict con metadatos extraídos por LLM
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        import os

        # Verificar API key
        api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
        if not api_key:
            log.warning("GOOGLE_API_KEY no configurada, saltando extracción con LLM")
            return {}

        # Inicializar LLM
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.1)

        # Muestrear texto
        sample = text[:config.llm_metadata_sample_size]

        prompt = f"""
        Extrae los siguientes metadatos del texto jurídico-sanitario.
        Si no encuentras algún dato, devuelve null.

        Texto:
        {sample}

        Devuelve SOLO un objeto JSON válido con esta estructura exacta:
        {{
            "autor": "nombre completo del autor o autores",
            "titulo": "título completo del documento",
            "anio": "año del documento en formato YYYY",
            "jurisdiccion": "provincia o jurisdicción",
            "tipo": "tipo de documento",
            "tribunal": "nombre del tribunal si aplica",
            "expediente": "número de expediente si existe"
        }}
        """

        response = llm.invoke(prompt)
        text_response = response.content if hasattr(response, "content") else str(response)

        # Extraer JSON
        import json
        json_match = re.search(r'\{.*\}', text_response, re.DOTALL)
        if json_match:
            metadata = json.loads(json_match.group(0))
            return metadata

        return {}

    except Exception as e:
        log.warning(f"Error en extracción con LLM: {e}")
        return {}


def detect_metadata_enriched(
    pdf_path: Path,
    text: str,
    config: IngestaConfig = DEFAULT_INGESTA_CONFIG
) -> MetadataEnriquecido:
    """
    Extrae metadatos del PDF y texto usando detección avanzada.

    Args:
        pdf_path: Ruta al PDF
        text: Texto extraído del PDF
        config: Configuración de ingesta

    Returns:
        MetadataEnriquecido con todos los campos detectados
    """
    # Hash del PDF
    with open(pdf_path, "rb") as f:
        h = hashlib.md5(f.read()).hexdigest()

    meta = MetadataEnriquecido(archivo_origen=pdf_path.name, hash_pdf=h)

    # === EXTRACCIÓN CON LLM (OPCIONAL) ===
    llm_metadata = {}
    if config.use_llm_metadata:
        log.info("Extrayendo metadata con Gemini...")
        llm_metadata = extract_metadata_with_llm(text, pdf_path, config)
        meta.metodo_deteccion = "llm"

    # === DETECCIÓN BÁSICA (SIEMPRE) ===
    meta.tipo_documento = classify_document(text)

    # Expediente
    mexp = re.search(_PATTERNS["expediente"], text)
    meta.expediente = llm_metadata.get("expediente") or (mexp.group(2) if mexp else None)

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
    meta.tribunal = llm_metadata.get("tribunal") or (mtrib.group(0) if mtrib else None)

    # === EXTRACCIÓN DE METADATA DEL PDF ===
    pdf_meta = extract_pdf_metadata(pdf_path)
    meta.num_paginas = pdf_meta["num_paginas"]

    # === DETECCIÓN AVANZADA (SEGÚN CONFIGURACIÓN) ===

    # Autor
    if config.extract_author:
        meta.autor = (
            llm_metadata.get("autor") or
            pdf_meta.get("autor_pdf") or
            detect_author_advanced(text) or
            ""
        )

    # Título
    if config.extract_title:
        meta.titulo = (
            llm_metadata.get("titulo") or
            pdf_meta.get("titulo_pdf") or
            detect_title_advanced(text) or
            ""
        )

    # Año
    if config.extract_year:
        meta.anio = (
            llm_metadata.get("anio") or
            detect_year_advanced(text)
        )

    # Jurisdicción
    if config.extract_jurisdiction:
        meta.jurisdiccion = (
            llm_metadata.get("jurisdiccion") or
            detect_jurisdiction_advanced(text) or
            ""
        )

    # Idioma
    if config.detect_language:
        meta.idioma = detect_language(text)

    # Log de resultados
    log.info(
        f"Metadata detectada: autor={bool(meta.autor)}, "
        f"titulo={bool(meta.titulo)}, año={meta.anio}, "
        f"jurisdiccion={bool(meta.jurisdiccion)}, idioma={meta.idioma}"
    )

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

    # 5. Metadata enriquecida (⭐ CON DETECCIÓN AVANZADA)
    meta_doc = detect_metadata_enriched(pdf_path, text, config)
    meta_doc.materia = materia or base_nombre
    meta_doc.fragmentos_extraidos = len(docs)

    if docs:
        meta_doc.score_promedio = sum(d.metadata.get("score_calidad", 0.5) for d in docs) / len(docs)
        meta_doc.fragmentos_validos = len(docs)

    # 6. Asignar metadata a cada fragmento (⭐ INCLUYE NUEVOS CAMPOS)
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
            "base_nombre": base_nombre,
            # ⭐ NUEVOS METADATOS
            "autor": meta_doc.autor,
            "titulo": meta_doc.titulo,
            "anio": meta_doc.anio,
            "jurisdiccion": meta_doc.jurisdiccion,
            "idioma": meta_doc.idioma,
            "url": meta_doc.url,
            "pagina": "1",  # Por defecto (cada fragmento viene de múltiples páginas)
            "num_paginas": meta_doc.num_paginas,
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
