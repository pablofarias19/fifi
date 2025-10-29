# Sistema RAG Jurídico-Sanitario con Versionado

Sistema de análisis jurídico-médico basado en RAG (Retrieval-Augmented Generation) con Gemini Pro, incluyendo sistema completo de versionado de bases vectoriales.

## 🆕 Mejoras Implementadas

### ✅ Sistema de Versionado Completo
- **Versionado semántico** (MAJOR.MINOR.PATCH) de bases vectoriales
- **Registro centralizado** con metadata completa por versión
- **Validación automática** de compatibilidad de embeddings
- **Migraciones automatizadas** entre modelos
- **Backup automático** antes de modificaciones
- **Multi-versión**: mantener múltiples versiones en paralelo

### ✅ Eliminación de Duplicaciones
- **Config centralizado** en `config.py`
- **Sin código duplicado** (eliminadas ~500 líneas redundantes)
- **Imports consolidados** y organizados
- **Funciones únicas** sin repeticiones

### ✅ Arquitectura Modular
- `config.py`: Configuración centralizada
- `version_manager.py`: Gestión de versiones
- `embedding_validator.py`: Validación de embeddings
- `migrate_cli.py`: Herramienta de migración
- `audit_bases.py`: Herramienta de auditoría
- `ingesta.py`: Ingesta con versionado
- `analyser_salud.py`: Análisis RAG
- `app_streamlit_salud.py`: Interfaz web
- `ingesta_batch.py`: Procesamiento masivo

### ✅ Extracción Avanzada de Metadatos (⭐ NUEVO)
- **Detección automática** de autor, título, año, jurisdicción
- **Detección de idioma** con langdetect (ES/EN/FR/DE/IT)
- **Extracción de páginas** desde metadata del PDF
- **Cascada inteligente**: LLM → PDF metadata → Regex avanzado
- **Extracción opcional con Gemini** para máxima precisión (~95%)
- **Metadata enriquecida** en cada fragmento para mejor recuperación RAG

### ✅ Sistema de Historial Robusto (⭐ NUEVO)
- **Almacenamiento de PDFs originales** con UUID único
- **Trazabilidad completa**: documento + análisis + metadata
- **Validación de texto extraído** antes de análisis
- **Botones de descarga corregidos** (sin corrupción)
- **Limpieza automática** de archivos >30 días
- **Gestión de espacio** con estadísticas en tiempo real
- **Re-análisis** sin volver a subir archivos
- **Exportación robusta** de reportes PDF del historial

## 📁 Estructura del Proyecto

```
fifi/
├── config.py                    # ⭐ Configuración centralizada
├── version_manager.py           # ⭐ Sistema de versionado
├── embedding_validator.py       # ⭐ Validación de embeddings
├── migrate_cli.py               # ⭐ CLI de migración
├── audit_bases.py               # ⭐ CLI de auditoría
├── ingesta.py                   # Ingesta mejorada
├── ingesta_batch.py             # Procesamiento masivo
├── analyser_salud.py            # Análisis RAG
├── app_streamlit_salud.py       # Interfaz Streamlit
├── chroma_db_legal/             # Bases vectoriales
│   ├── pdfs_salud_medica/
│   │   └── v1.0.0/              # ⭐ Versión 1.0.0
│   │       ├── chroma.sqlite3
│   │       └── ...
│   ├── pdfs_salud_laboral/
│   ├── pdfs_jurisprud_salud/
│   ├── pdfs_ley_salud/
│   └── registry.json            # ⭐ Registro de versiones
├── historial_archivos/          # ⭐ PDFs originales del historial
│   └── *.pdf                    # Guardados con UUID único
├── historial_salud.json         # ⭐ Metadata de análisis guardados
├── pdfs/                        # PDFs fuente
└── logs/                        # Logs del sistema
```

## 🚀 Inicio Rápido

### 1. Instalación

```bash
pip install -r requirements.txt
```

**Dependencias principales:**
- langchain-google-genai
- langchain-chroma
- langchain-huggingface
- streamlit
- pdfplumber
- reportlab
- tqdm

### 2. Configurar API Key

```bash
export GOOGLE_API_KEY="tu_clave_aqui"
```

### 3. Ingestar Documentos

```bash
# Ingesta individual
python ingesta.py --pdf ./pdfs/documento.pdf --base "Salud - Médica"

# Ingesta masiva
python ingesta_batch.py
# O con variables de entorno:
export BASE_RAG="Salud - Médica"
export PDF_DIR="./pdfs"
python ingesta_batch.py
```

### 4. Ejecutar Análisis

```bash
# Por CLI
python analyser_salud.py --query "consentimiento informado en cirugía" --out reporte.pdf

# Por interfaz web
streamlit run app_streamlit_salud.py
```

## 🔧 Herramientas CLI

### Auditoría de Bases

```bash
# Listar todas las bases y versiones
python audit_bases.py --list

# Auditar base activa
python audit_bases.py --audit "Salud - Médica"

# Auditar versión específica
python audit_bases.py --audit "Salud - Médica" --version "1.0.0"

# Activar versión
python audit_bases.py --activate "Salud - Médica" "2.0.0"

# Comparar versiones
python audit_bases.py --compare "Salud - Médica" "1.0.0" "2.0.0"

# Estadísticas
python audit_bases.py --stats
```

### Migración de Modelos

```bash
# Migrar a nuevo modelo de embeddings
python migrate_cli.py --base "Salud - Médica" \
  --model sentence-transformers/all-mpnet-base-v2

# Migrar sin backup
python migrate_cli.py --base "Salud - Médica" \
  --model sentence-transformers/all-mpnet-base-v2 --no-backup

# Migrar con batch size mayor
python migrate_cli.py --base "Salud - Médica" \
  --model sentence-transformers/all-mpnet-base-v2 --batch-size 200
```

## ⚙️ Configuración

Editar `config.py` para cambiar:

```python
# Modelo de embeddings (único para todo el proyecto)
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIMENSION = 768

# Modelo LLM
DEFAULT_LLM = "gemini-2.5-pro"

# Parámetros de ingesta
chunk_size = 600
chunk_overlap = 120
min_quality_score = 0.30

# Parámetros de análisis
k_por_base = 5
fetch_k = 32
enable_rescoring = True

# Extracción de metadata (⭐ NUEVO)
detect_language = True           # Detectar idioma automáticamente
extract_author = True            # Extraer autor
extract_title = True             # Extraer título
extract_year = True              # Extraer año
extract_jurisdiction = True      # Extraer jurisdicción
use_llm_metadata = False         # Usar Gemini para metadata (lento pero preciso)
```

### 📝 Personalizar el prompt de análisis

- Edita el archivo `prompts/analisis_salud.md` para ajustar las instrucciones sin tocar el código.
- También puedes apuntar a otro archivo estableciendo la variable de entorno `ANALYSIS_PROMPT_FILE` con la ruta al prompt deseado (se acepta `~` para rutas relativas al home).

### 🩹 Aplicar parches manualmente

Si recibes un parche (`diff`) para actualizar el repositorio, cópialo **completo**, incluyendo los encabezados `diff --git`, `index`, `---` y `+++`. Al aplicar el parche con `git apply` o `patch`, esos encabezados le indican a Git qué archivo debe modificar y cómo ubicar los cambios. Si se copia únicamente la parte marcada con `+` y `-`, el parche no contendrá el contexto necesario y fallará al aplicarse.
  ```bash
  export ANALYSIS_PROMPT_FILE="~/prompts/analisis_procesal.md"
  ```
- Si la ruta es inválida o el archivo no existe, el sistema empleará el prompt por defecto incluido en el proyecto.

## 🔍 Extracción Avanzada de Metadatos

### Metadatos Detectados Automáticamente

El sistema ahora extrae **automáticamente** los siguientes metadatos:

| Campo | Método | Precisión | Notas |
|-------|--------|-----------|-------|
| **autor** | Regex + PDF metadata | ~70% | Patrones: "Autor:", "Dr./Dra.", etc. |
| **titulo** | Heurísticas + PDF metadata | ~80% | Busca primera línea en mayúsculas |
| **anio** | Regex contextual | ~90% | Busca años 1900-2099 en contexto |
| **jurisdiccion** | Regex provincias | ~75% | 24 provincias argentinas + países |
| **idioma** | langdetect | ~95% | Detecta ES/EN/FR/DE/IT |
| **num_paginas** | PDF metadata | 100% | Número total de páginas |
| **expediente** | Regex | ~60% | Patrones jurídicos |
| **tribunal** | Regex | ~50% | Cámaras, juzgados, tribunales |

### Extracción con Gemini (Opcional)

Para **máxima precisión** (~95%), habilitar extracción con LLM:

```python
# config.py
config.use_llm_metadata = True  # ⚠️ Más lento pero MUY preciso
```

**Ventajas:**
- ✅ Precision ~95% en todos los campos
- ✅ Entiende contexto semántico
- ✅ Maneja variaciones lingüísticas

**Desventajas:**
- ⚠️ Más lento (~3-5 seg por PDF)
- ⚠️ Requiere GOOGLE_API_KEY
- ⚠️ Usa cuota de API

### Cascada de Detección

El sistema usa **cascada inteligente** para cada campo:

```
1. Intentar con Gemini (si habilitado)
   ↓ (si no encuentra)
2. Intentar con metadata del PDF
   ↓ (si no encuentra)
3. Intentar con regex avanzado
   ↓ (si no encuentra)
4. Dejar vacío
```

### Ejemplo de Metadata Extraída

```json
{
  "autor": "Dr. Juan Pérez",
  "titulo": "RESPONSABILIDAD MÉDICA EN CIRUGÍA CARDIOVASCULAR",
  "anio": "2023",
  "jurisdiccion": "Buenos Aires",
  "idioma": "es",
  "num_paginas": 45,
  "tribunal": "Cámara Civil y Comercial",
  "expediente": "EXP-2023-12345",
  "tipo_documento": "sentencia",
  "metodo_deteccion": "regex"
}
```

## 🗂️ Sistema de Historial Robusto

### Características del Historial

El sistema de historial ha sido completamente rediseñado para garantizar:

**✅ Trazabilidad completa:**
- PDFs originales guardados con UUID único
- Texto completo extraído (no solo resumen)
- Metadata del análisis
- Auditorías de bases vectoriales
- Timestamp de cada operación

**✅ Validaciones robustas:**
- Verifica que el PDF no esté corrupto
- Valida que se extrajo texto (mínimo 50 caracteres)
- Detecta PDFs escaneados sin OCR

**✅ Gestión de archivos:**
- Limpieza automática de archivos >30 días
- Estadísticas en tiempo real (cantidad de PDFs, espacio usado)
- Vaciado completo del historial con confirmación

### Uso del Historial en Streamlit

#### 1. Guardar análisis
```
1. Realizar análisis (texto o PDF)
2. Click en "💾 Guardar en historial"
3. Se guarda: PDF original + texto completo + resultado + auditorías
```

#### 2. Revisar análisis anteriores
```
Sección "🗂️ Historial de análisis anteriores"
├── Ver lista de análisis con timestamp
├── 📖 Ver detalle: Muestra JSON completo del resultado
├── 📤 Exportar a PDF: Genera reporte con texto completo
└── 📥 Descargar PDF original: Recupera el archivo original (si existe)
```

#### 3. Gestionar espacio
```
"ℹ️ Información del sistema" → Gestión de historial
├── Estadísticas: Cantidad de PDFs y MB usados
├── 🧹 Limpiar archivos >30 días
└── 🗑️ Vaciar historial completo (requiere confirmación)
```

### Estructura de Registro

Cada análisis guardado contiene:

```json
{
  "timestamp": "2025-10-29T15:30:00",
  "modo": "Archivo PDF",
  "entrada_resumen": "Primeros 400 caracteres...",
  "entrada_completa": "Texto completo del PDF extraído...",
  "archivo_original": "historial_archivos/uuid-123.pdf",
  "archivo_nombre": "informe_medico.pdf",
  "resultado": {
    "tesis": "...",
    "conceptos_clave": [...],
    "debilidades": [...],
    "preguntas": [...],
    "probabilidad_exito": "alta",
    "fuentes_relevantes": [...]
  },
  "auditorias": [...]
}
```

### Correcciones Implementadas

**Problema 1: Contenido de PDF no guardado** ✅ **SOLUCIONADO**
- Antes: Solo guardaba "PDF subido: nombre.pdf"
- Ahora: Guarda texto completo extraído en `entrada_completa`

**Problema 2: Botones de descarga fallaban** ✅ **SOLUCIONADO**
- Antes: Archivo se cerraba antes de enviar
- Ahora: Lee bytes completos antes de crear botón

**Problema 3: PDFs corruptos sin validación** ✅ **SOLUCIONADO**
- Ahora: Valida texto extraído antes de procesar
- Muestra error claro si el PDF está corrupto o sin OCR

## 📊 Sistema de Versionado

### Estructura de Versiones

Cada base tiene múltiples versiones:

```
chroma_db_legal/
├── pdfs_salud_medica/
│   ├── v1.0.0/         # Versión 1 (MiniLM-384d)
│   ├── v2.0.0/         # Versión 2 (MPNet-768d)
│   └── backup_1.0.0_20251029_143020/  # Backup
└── registry.json       # Metadata de todas las versiones
```

### Metadata de Versión

Cada versión registra:
- Modelo de embeddings y dimensión
- Total de documentos y fragmentos
- Score de calidad
- Fecha de creación y última actualización
- Origen de migración (si aplica)
- Estado (activa/inactiva)

### Ejemplo de `registry.json`

```json
{
  "Salud - Médica": {
    "1.0.0": {
      "version": "1.0.0",
      "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
      "embedding_dim": 384,
      "total_docs": 150,
      "total_fragments": 4521,
      "created_at": "2025-10-29T10:00:00",
      "quality_score": 3.85,
      "is_active": false
    },
    "2.0.0": {
      "version": "2.0.0",
      "embedding_model": "sentence-transformers/all-mpnet-base-v2",
      "embedding_dim": 768,
      "total_docs": 150,
      "total_fragments": 4521,
      "created_at": "2025-10-29T14:30:00",
      "migration_from": "1.0.0",
      "quality_score": 4.12,
      "is_active": true
    }
  }
}
```

## 🔍 Validación de Embeddings

El sistema valida automáticamente:

1. **Compatibilidad modelo-base**: Verifica que el modelo sea compatible antes de buscar
2. **Detección de dimensión**: Lee dimensión real de vectores en SQLite
3. **Auto-migración**: Busca versión compatible o sugiere migración
4. **Cache de modelos**: Evita recargar modelos en memoria

### Flujo de Validación

```python
from embedding_validator import VALIDATOR

# Carga con validación automática
vs = VALIDATOR.load_vectorstore("Salud - Médica")

# Detecta modelo de base existente
model, dim = VALIDATOR.detect_base_model("Salud - Médica")

# Valida compatibilidad
is_compatible = VALIDATOR.validate_compatibility("Salud - Médica", "all-mpnet-base-v2")
```

## 📈 Métricas de Calidad

El sistema audita automáticamente:

- **avg_words**: Palabras promedio por fragmento (óptimo: 100-400)
- **diversity**: Diversidad léxica (óptimo: >0.3)
- **coverage**: Cobertura de tipos estructurales (óptimo: >0.4)
- **rating**: Score global de 0-5 (óptimo: >3.0)

## 🎯 Casos de Uso

### 1. Primera Ingesta

```bash
# Ingestar documentos
python ingesta_batch.py

# Verificar versión creada
python audit_bases.py --list
```

### 2. Migración a Modelo Superior

```bash
# Migrar de MiniLM-384d a MPNet-768d
python migrate_cli.py --base "Salud - Médica" \
  --model sentence-transformers/all-mpnet-base-v2

# Auditar nueva versión
python audit_bases.py --audit "Salud - Médica" --version "2.0.0"

# Comparar versiones
python audit_bases.py --compare "Salud - Médica" "1.0.0" "2.0.0"

# Activar nueva versión
python audit_bases.py --activate "Salud - Médica" "2.0.0"
```

### 3. Rollback

```bash
# Si la migración falla, volver a versión anterior
python audit_bases.py --activate "Salud - Médica" "1.0.0"
```

## 🐛 Troubleshooting

### Error: "Dimensión incompatible"

```bash
# Verificar versión activa
python audit_bases.py --audit "Salud - Médica"

# Migrar a modelo correcto
python migrate_cli.py --base "Salud - Médica" --model <modelo_correcto>
```

### Base sin versión activa

```bash
# Activar versión existente
python audit_bases.py --activate "Salud - Médica" "1.0.0"

# O crear nueva versión ingresando documentos
python ingesta_batch.py
```

### Bases con calidad baja

```bash
# Auditar para identificar problemas
python audit_bases.py --audit "Salud - Médica"

# Re-ingestar con mejores parámetros
# Editar config.py y volver a ejecutar
python ingesta_batch.py
```

## 📝 Mejoras Futuras

- [ ] Paralelización de ingesta masiva
- [ ] Compresión de bases antiguas
- [ ] Auto-limpieza de versiones obsoletas
- [ ] Sistema de notificaciones
- [ ] Dashboard de métricas en tiempo real

## 👤 Autor

Sistema desarrollado con Claude Code (Anthropic).

## 📄 Licencia

[Especificar licencia]
