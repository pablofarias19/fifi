# -*- coding: utf-8 -*-
"""
Ingesta masiva mejorada (jurídico-sanitaria)
--------------------------------------------
Procesa todos los PDFs de una carpeta, con modelo y base configurables.
"""

import os
import subprocess
from pathlib import Path

# ======================
# Configuración
# ======================
PDF_DIR = Path(__file__).parent / "pdfs"
BASE_RAG = os.getenv("BASE_RAG", "Salud - Médica")
MATERIA = os.getenv("MATERIA", None)
MODEL = os.getenv("MODEL", "sentence-transformers/all-mpnet-base-v2")  # 768d
USE_OCR = os.getenv("USE_OCR", "1") == "1"  # habilitado por defecto

# ======================
# Ejecución
# ======================
for pdf_file in PDF_DIR.glob("*.pdf"):
    if not pdf_file.is_file():
        continue
    cmd = [
        "python", "ingesta.py",
        "--pdf", str(pdf_file),
        "--base", BASE_RAG,
        "--model", MODEL
    ]
    if not USE_OCR:
        cmd.append("--no-ocr")
    if MATERIA:
        cmd += ["--materia", MATERIA]

    print(f"📄 Procesando: {pdf_file.name} → {BASE_RAG} ({MODEL})")
    subprocess.run(cmd, check=False)

print("\n✅ Ingesta masiva finalizada.")
