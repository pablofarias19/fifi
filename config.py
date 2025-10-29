diff --git a/config.py b/config.py
index 4aeab70f77ea23307c4edc1c30cf9b67a38f05e4..d7b2c06c7c8f1233ce605c64f08344db77b68fa1 100644
--- a/config.py
+++ b/config.py
@@ -1,85 +1,172 @@
 # -*- coding: utf-8 -*-
 """
 CONFIGURACIÓN CENTRALIZADA - Sistema RAG Jurídico-Sanitario
 ------------------------------------------------------------
 Configuración única para embeddings, rutas, modelos y parámetros.
 Elimina duplicaciones y asegura consistencia en todo el proyecto.
 """
 
-from pathlib import Path
+import os
+from pathlib import Path
 from dataclasses import dataclass, field
 from typing import Dict
 
 # ======================
 # RUTAS PRINCIPALES
 # ======================
 
-ROOT = Path(__file__).parent
-PDF_DIR_DEFAULT = ROOT / "pdfs"
-DB_DIR_DEFAULT = ROOT / "chroma_db_legal"
-CACHE_DIR_DEFAULT = ROOT / "chroma_cache"
-LOGS_DIR = ROOT / "logs"
-REGISTRY_PATH = DB_DIR_DEFAULT / "registry.json"
+ROOT = Path(__file__).parent
+PDF_DIR_DEFAULT = ROOT / "pdfs"
+DB_DIR_DEFAULT = ROOT / "chroma_db_legal"
+CACHE_DIR_DEFAULT = ROOT / "chroma_cache"
+LOGS_DIR = ROOT / "logs"
+PROMPTS_DIR = ROOT / "prompts"
+REGISTRY_PATH = DB_DIR_DEFAULT / "registry.json"
 
 # Asegurar que existen
-for d in [PDF_DIR_DEFAULT, DB_DIR_DEFAULT, CACHE_DIR_DEFAULT, LOGS_DIR]:
-    d.mkdir(exist_ok=True)
+for d in [PDF_DIR_DEFAULT, DB_DIR_DEFAULT, CACHE_DIR_DEFAULT, LOGS_DIR, PROMPTS_DIR]:
+    d.mkdir(exist_ok=True)
 
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
 
-DEFAULT_LLM = "gemini-2.5-pro"
-DEFAULT_TEMPERATURE = 0.2
+DEFAULT_LLM = "gemini-2.5-pro"
+DEFAULT_TEMPERATURE = 0.2
+
+# ======================
+# PROMPTS DE ANÁLISIS
+# ======================
+
+PROMPT_ANALISIS_PATH = PROMPTS_DIR / "analisis_salud.md"
+
+DEFAULT_ANALYSIS_PROMPT = """[Contexto recuperado con metadatos y referencias]
+{contexto}
+
+[Texto o consulta principal]
+{texto}
+
+[INSTRUCCIONES DE ANÁLISIS]
+1️⃣ **Síntesis contextual**
+   - Resume el caso y su contexto jurídico-médico.
+   - Determina tipo de responsabilidad (civil, penal, laboral, bioética).
+   - Reconoce las fuentes primarias y secundarias relevantes.
+
+2️⃣ **Ejes conceptuales**
+   - Desarrolla los temas: consentimiento informado, lex artis, causalidad, daño, prueba.
+   - Para cada eje: explica el concepto, cita fuentes con autor/año/página o tribunal/año.
+   - Usa las referencias documentales para sostener los razonamientos.
+
+3️⃣ **Evaluación argumental**
+   - Señala los argumentos más sólidos y las debilidades.
+   - Indica contradicciones entre jurisprudencias o doctrinas.
+   - Identifica lagunas normativas o probatorias.
+
+4️⃣ **Fuentes y trazabilidad**
+   - Crea una lista de fuentes relevantes con formato:
+     "Autor – Título (Año), pág. X [URL o Jurisdicción]"
+   - Indica si son doctrina, jurisprudencia o normativa.
+   - Recomienda cuáles podrían ampliarse con consultas específicas o documentos adicionales.
+
+5️⃣ **Preguntas derivadas**
+   - Formula 3 a 5 preguntas que profundicen el análisis.
+   - Indica qué información o tipo de documento sería necesario obtener para resolverlas.
+
+6️⃣ **Conclusión jurídica**
+   - Formula una tesis integrada, valorando la probabilidad de éxito (baja/media/alta).
+   - Justifica con criterios jurídicos y médicos.
+
+[FORMATO DE SALIDA]
+Primero entrega un informe narrativo completo con secciones jerarquizadas:
+🔹 Resumen contextual
+🔹 Ejes conceptuales
+🔹 Debilidades / Vacíos argumentales
+🔹 Preguntas de profundización
+🔹 Conclusión jurídica y fuentes
+
+Luego, entrega un bloque JSON resumido con esta estructura:
+
+{{
+  "tesis": "...",
+  "conceptos_clave": ["..."],
+  "debilidades": ["..."],
+  "preguntas": ["..."],
+  "probabilidad_exito": "baja|media|alta",
+  "fuentes_relevantes": [
+    {{"autor": "...", "titulo": "...", "anio": "...", "pagina": "...", "url": "...", "tipo": "..."}}
+  ]
+}}
+"""
+
+
+def get_analysis_prompt(path: Path = PROMPT_ANALISIS_PATH, default: str = DEFAULT_ANALYSIS_PROMPT) -> str:
+    """Retorna el prompt de análisis, permitiendo sobrescribirlo mediante archivo."""
+
+    # Permite sobreescribir la ruta vía variable de entorno.
+    env_override = os.environ.get("ANALYSIS_PROMPT_FILE")
+    candidates = []
+    if env_override:
+        candidates.append(Path(env_override).expanduser())
+    if path:
+        candidates.append(path)
+
+    for candidate in candidates:
+        try:
+            if candidate.exists():
+                return candidate.read_text(encoding="utf-8")
+        except OSError:
+            # Ignora rutas inválidas y continúa con el siguiente candidato.
+            continue
+    return default
 
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
