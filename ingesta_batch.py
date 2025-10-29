# -*- coding: utf-8 -*-
"""
Ingesta masiva mejorada (jurídico-sanitaria)
--------------------------------------------
Procesa todos los PDFs de una carpeta con versionado automático.
Integrado con el sistema de configuración centralizado.
"""

import os, sys
from pathlib import Path
from typing import Optional

from config import PDF_DIR_DEFAULT, BASES_RAG, DEFAULT_INGESTA_CONFIG
from ingesta import ingest_pdf_to_chroma
from version_manager import REGISTRY

# ======================
# Configuración desde variables de entorno
# ======================

PDF_DIR = Path(os.getenv("PDF_DIR", str(PDF_DIR_DEFAULT)))
BASE_RAG = os.getenv("BASE_RAG", "Salud - Médica")
MATERIA = os.getenv("MATERIA", None)
USE_OCR = os.getenv("USE_OCR", "1") == "1"

# ======================
# Validación
# ======================

if BASE_RAG not in BASES_RAG:
    print(f"❌ Error: BASE_RAG '{BASE_RAG}' no es válida")
    print(f"   Bases disponibles: {', '.join(BASES_RAG.keys())}")
    sys.exit(1)

if not PDF_DIR.exists():
    print(f"❌ Error: Directorio de PDFs no existe: {PDF_DIR}")
    sys.exit(1)

# ======================
# Búsqueda de PDFs
# ======================

pdf_files = list(PDF_DIR.glob("*.pdf"))

if not pdf_files:
    print(f"⚠️  No se encontraron archivos PDF en {PDF_DIR}")
    sys.exit(0)

# ======================
# Procesamiento
# ======================

print(f"\n🔄 INGESTA MASIVA")
print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"Directorio: {PDF_DIR}")
print(f"Base destino: {BASE_RAG}")
print(f"Materia: {MATERIA or '(sin especificar)'}")
print(f"OCR: {'Habilitado' if USE_OCR else 'Deshabilitado'}")
print(f"Archivos encontrados: {len(pdf_files)}")

# Configurar ingesta
config = DEFAULT_INGESTA_CONFIG
config.usar_ocr = USE_OCR

# Estadísticas
total = len(pdf_files)
exitosos = 0
fallidos = 0
total_fragmentos = 0

print(f"\n⚙️  Procesando PDFs...")
print("─" * 70)

for i, pdf_file in enumerate(pdf_files, 1):
    if not pdf_file.is_file():
        continue

    print(f"\n[{i}/{total}] 📄 {pdf_file.name}")

    try:
        result = ingest_pdf_to_chroma(
            pdf_path=pdf_file,
            base_nombre=BASE_RAG,
            materia=MATERIA,
            config=config
        )

        if result.get("ok", False):
            exitosos += 1
            frags = result.get("n_fragmentos", 0)
            total_fragmentos += frags
            version = result.get("version", "?")
            print(f"   ✅ Exitoso: {frags} fragmentos → v{version}")
        else:
            fallidos += 1
            error = result.get("error", "Error desconocido")
            print(f"   ❌ Falló: {error}")

    except Exception as e:
        fallidos += 1
        print(f"   ❌ Error: {e}")

# ======================
# Resumen final
# ======================

print(f"\n🎉 INGESTA COMPLETADA")
print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"Base: {BASE_RAG}")
print(f"Total procesados: {total}")
print(f"Exitosos: {exitosos} ({exitosos/total*100:.1f}%)" if total > 0 else "Exitosos: 0")
print(f"Fallidos: {fallidos} ({fallidos/total*100:.1f}%)" if total > 0 else "Fallidos: 0")
print(f"Fragmentos totales ingresados: {total_fragmentos}")

# Mostrar información de versión
active_version = REGISTRY.get_active_version(BASE_RAG)
if active_version:
    meta = REGISTRY.get_version(BASE_RAG, active_version)
    if meta:
        print(f"\n📊 Versión activa: v{active_version}")
        print(f"   Modelo: {meta.embedding_model} ({meta.embedding_dim}d)")
        print(f"   Documentos: {meta.total_docs}")
        print(f"   Fragmentos: {meta.total_fragments}")

sys.exit(0 if fallidos == 0 else 1)
