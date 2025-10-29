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
```

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
