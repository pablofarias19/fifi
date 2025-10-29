# -*- coding: utf-8 -*-
"""
CONFIGURACIÓN CENTRALIZADA - Sistema RAG Jurídico-Sanitario
------------------------------------------------------------
Configuración única para embeddings, rutas, modelos y parámetros.
Elimina duplicaciones y asegura consistencia en todo el proyecto.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict

# ======================
# RUTAS PRINCIPALES
# ======================

ROOT = Path(__file__).parent
PDF_DIR_DEFAULT = ROOT / "pdfs"
DB_DIR_DEFAULT = ROOT / "chroma_db_legal"
CACHE_DIR_DEFAULT = ROOT / "chroma_cache"
LOGS_DIR = ROOT / "logs"
REGISTRY_PATH = DB_DIR_DEFAULT / "registry.json"

# Asegurar que existen
for d in [PDF_DIR_DEFAULT, DB_DIR_DEFAULT, CACHE_DIR_DEFAULT, LOGS_DIR]:
    d.mkdir(exist_ok=True)

# ======================
# EMBEDDINGS - CONFIGURACIÓN ÚNICA
# ======================

# Modelo estándar para TODO el proyecto
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIMENSION = 768

# Modelos soportados (para migraciones)
SUPPORTED_MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": 768,
}

# ======================
# BASES RAG DISPONIBLES
# ======================

BASES_RAG: Dict[str, str] = {
    "Salud - Médica": "pdfs_salud_medica",
    "Salud - Laboral": "pdfs_salud_laboral",
    "Jurisprudencia - Salud": "pdfs_jurisprud_salud",
    "Legislación - Salud": "pdfs_ley_salud",
}

# ======================
# LLM (GEMINI)
# ======================

DEFAULT_LLM = "gemini-2.5-pro"
DEFAULT_TEMPERATURE = 0.2

# ======================
# OCR
# ======================

IDIOMAS_VALIDOS = ["en", "de", "es", "fr", "it"]
LANG_TESSERACT = "spa+eng+fra+ita+deu"

# ======================
# PARÁMETROS DE INGESTA
# ======================

@dataclass
class IngestaConfig:
    """Configuración para ingesta de documentos"""
    chunk_size: int = 600
    chunk_overlap: int = 120
    usar_ocr: bool = True
    dpi_ocr: int = 300
    lang_ocr: str = LANG_TESSERACT
    min_quality_score: float = 0.30

    # Nuevos parámetros de versionado
    auto_version: bool = True  # Auto-incrementar versión en cambios
    create_backup: bool = True  # Backup antes de modificar

# ======================
# PARÁMETROS DE ANÁLISIS
# ======================

@dataclass
class AnalysisConfig:
    """Configuración para análisis RAG"""
    k_por_base: int = 5           # Fragmentos a recuperar por base
    fetch_k: int = 32              # Candidatos para re-ranking
    max_context_length: int = 8000 # Longitud máxima de contexto

    # Re-scoring
    enable_rescoring: bool = True
    keyword_bonus_max: float = 0.20
    structure_bonus: float = 0.05
    quality_bonus_max: float = 0.10

# ======================
# KEYWORDS JURÍDICO-SANITARIAS
# ======================

KEYWORDS_JURIDICO_SANIT = (
    "artículo", "articulo", "considerando", "resuelve", "sentencia", "fallo",
    "tribunal", "cámara", "juzgado", "demanda", "dictamen",
    "consentimiento informado", "historia clínica", "lex artis", "mala praxis",
    "paciente", "responsabilidad", "causalidad", "daño", "negligencia",
    "protocolo", "bioética", "OMS", "OPS"
)

# ======================
# INSTANCIAS POR DEFECTO
# ======================

DEFAULT_INGESTA_CONFIG = IngestaConfig()
DEFAULT_ANALYSIS_CONFIG = AnalysisConfig()

# ======================
# SISTEMA DE VERSIONADO
# ======================

VERSION_SYSTEM = {
    "enabled": True,
    "semantic_versioning": True,  # Usa versionado semántico (MAJOR.MINOR.PATCH)
    "keep_old_versions": True,     # Mantiene versiones antiguas
    "max_versions_per_base": 3,    # Máximo de versiones a mantener
}
