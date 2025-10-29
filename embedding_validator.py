# -*- coding: utf-8 -*-
"""
EMBEDDING VALIDATOR - Validación y Gestión de Compatibilidad
-------------------------------------------------------------
Valida embeddings, detecta dimensiones y gestiona carga segura.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict

from langchain_huggingface import HuggingFaceEmbeddings as SentenceTransformerEmbeddings
from langchain_chroma import Chroma

from config import EMBEDDING_MODEL, EMBEDDING_DIMENSION, SUPPORTED_MODELS
from version_manager import REGISTRY, get_version_directory

# ======================
# Logger
# ======================

log = logging.getLogger("embedding_validator")
log.setLevel(logging.INFO)
if not log.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)

# ======================
# Excepciones
# ======================

class IncompatibleEmbeddingError(Exception):
    """Error cuando embeddings no coinciden con base"""
    pass


class EmbeddingNotFoundError(Exception):
    """Error cuando no se encuentra modelo de embeddings"""
    pass


# ======================
# Validador Principal
# ======================

class EmbeddingValidator:
    """
    Valida compatibilidad de embeddings antes de usar bases vectoriales.
    Proporciona carga segura y sugerencias de migración.
    """

    def __init__(self):
        self.registry = REGISTRY
        self._embedding_cache: Dict[str, SentenceTransformerEmbeddings] = {}

    # ======================
    # Carga de Embeddings
    # ======================

    def get_embeddings(self, model_name: str = EMBEDDING_MODEL) -> SentenceTransformerEmbeddings:
        """
        Obtiene instancia de embeddings, con cache para reusar.

        Args:
            model_name: Nombre del modelo HuggingFace

        Returns:
            Instancia de SentenceTransformerEmbeddings
        """
        if model_name not in self._embedding_cache:
            log.info(f"🧠 Cargando modelo de embeddings: {model_name}")
            try:
                embeddings = SentenceTransformerEmbeddings(model_name=model_name)
                self._embedding_cache[model_name] = embeddings
                log.info(f"✅ Modelo cargado exitosamente")
            except Exception as e:
                log.error(f"❌ Error cargando modelo {model_name}: {e}")
                raise EmbeddingNotFoundError(f"No se pudo cargar {model_name}: {e}")

        return self._embedding_cache[model_name]

    def get_embedding_dimension(self, model_name: str) -> Optional[int]:
        """
        Obtiene dimensión de un modelo de embeddings.

        Args:
            model_name: Nombre del modelo

        Returns:
            Dimensión del vector o None si no se conoce
        """
        # Primero buscar en modelos soportados
        if model_name in SUPPORTED_MODELS:
            return SUPPORTED_MODELS[model_name]

        # Intentar cargar y detectar
        try:
            embeddings = self.get_embeddings(model_name)

            # Intentar obtener dimensión del modelo
            if hasattr(embeddings, "client") and hasattr(embeddings.client, "get_sentence_embedding_dimension"):
                return embeddings.client.get_sentence_embedding_dimension()

            # Método alternativo: generar embedding de prueba
            test_vec = embeddings.embed_query("test")
            return len(test_vec)

        except Exception as e:
            log.warning(f"No se pudo detectar dimensión de {model_name}: {e}")
            return None

    # ======================
    # Detección de Base Existente
    # ======================

    def detect_base_model(self, base_name: str) -> Optional[Tuple[str, int]]:
        """
        Detecta modelo y dimensión de una base existente.

        Args:
            base_name: Nombre de la base

        Returns:
            Tupla (modelo, dimensión) o None
        """
        active_version = self.registry.get_active_version(base_name)
        if not active_version:
            log.warning(f"No hay versión activa para {base_name}")
            return None

        version_data = self.registry.get_version(base_name, active_version)
        if not version_data:
            return None

        return (version_data.embedding_model, version_data.embedding_dim)

    def detect_from_directory(self, version_dir: Path) -> Optional[Tuple[str, int]]:
        """
        Detecta modelo leyendo vectores directamente de Chroma SQLite.

        Args:
            version_dir: Directorio de la versión

        Returns:
            Tupla (modelo, dimensión) o None
        """
        db_file = version_dir / "chroma.sqlite3"
        if not db_file.exists():
            log.warning(f"No existe base Chroma en {version_dir}")
            return None

        try:
            import sqlite3
            import numpy as np

            conn = sqlite3.connect(db_file)
            cur = conn.cursor()

            # Intentar leer un embedding
            cur.execute("SELECT embedding FROM embeddings LIMIT 1")
            row = cur.fetchone()
            conn.close()

            if row and row[0]:
                # Deserializar vector
                vector = np.frombuffer(row[0], dtype=np.float32)
                dimension = len(vector)

                # Mapear dimensión a modelo común
                for model, dim in SUPPORTED_MODELS.items():
                    if dim == dimension:
                        log.info(f"🔍 Detectado: {model} ({dimension}d) en {version_dir.name}")
                        return (model, dimension)

                log.warning(f"Dimensión {dimension} no corresponde a modelos conocidos")
                return (None, dimension)

        except Exception as e:
            log.error(f"Error detectando desde directorio: {e}")
            return None

    # ======================
    # Validación
    # ======================

    def validate_compatibility(
        self,
        base_name: str,
        model_name: str = EMBEDDING_MODEL
    ) -> bool:
        """
        Valida si un modelo es compatible con una base.

        Args:
            base_name: Nombre de la base
            model_name: Modelo a validar

        Returns:
            True si compatible, False si no
        """
        detected = self.detect_base_model(base_name)
        if not detected:
            log.warning(f"No se pudo detectar modelo de {base_name}")
            return False

        base_model, base_dim = detected
        target_dim = self.get_embedding_dimension(model_name)

        if base_model == model_name:
            log.info(f"✅ Modelo compatible: {model_name}")
            return True

        if base_dim == target_dim:
            log.warning(
                f"⚠️ Modelos diferentes pero dimensión compatible: "
                f"{base_model} ({base_dim}d) vs {model_name} ({target_dim}d)"
            )
            return True

        log.error(
            f"❌ Incompatible: base usa {base_model} ({base_dim}d), "
            f"intentas usar {model_name} ({target_dim}d)"
        )
        return False

    # ======================
    # Carga Segura de Vectorstore
    # ======================

    def load_vectorstore(
        self,
        base_name: str,
        model_name: str = EMBEDDING_MODEL,
        auto_migrate: bool = False
    ) -> Optional[Chroma]:
        """
        Carga vectorstore con validación de compatibilidad.

        Args:
            base_name: Nombre de la base
            model_name: Modelo de embeddings a usar
            auto_migrate: Si True, intenta encontrar versión compatible

        Returns:
            Instancia de Chroma o None si falla

        Raises:
            IncompatibleEmbeddingError: Si embeddings incompatibles
        """
        # 1. Validar compatibilidad
        if not self.validate_compatibility(base_name, model_name):
            if auto_migrate:
                # Buscar versión compatible
                compatible_version = self.registry.get_compatible_version(base_name, model_name)
                if compatible_version:
                    log.info(f"🔄 Usando versión compatible: {compatible_version}")
                    active_version = compatible_version
                else:
                    raise IncompatibleEmbeddingError(
                        f"Base {base_name} incompatible con {model_name}. "
                        f"Ejecuta: python migrate_cli.py --base '{base_name}' --model '{model_name}'"
                    )
            else:
                raise IncompatibleEmbeddingError(
                    f"Base {base_name} incompatible con {model_name}"
                )
        else:
            active_version = self.registry.get_active_version(base_name)

        # 2. Obtener directorio de versión
        version_dir = get_version_directory(base_name, active_version)
        if not version_dir.exists():
            log.error(f"Directorio no existe: {version_dir}")
            return None

        # 3. Cargar embeddings
        embeddings = self.get_embeddings(model_name)

        # 4. Crear instancia Chroma
        try:
            vs = Chroma(
                collection_name="legal_fragments",
                embedding_function=embeddings,
                persist_directory=str(version_dir)
            )
            log.info(f"✅ Vectorstore cargado: {base_name} v{active_version}")
            return vs

        except Exception as e:
            log.error(f"Error cargando vectorstore: {e}")
            return None

    def load_all_vectorstores(
        self,
        bases: Dict[str, str],
        model_name: str = EMBEDDING_MODEL
    ) -> Dict[str, Chroma]:
        """
        Carga múltiples vectorstores con validación.

        Args:
            bases: Diccionario de bases (nombre -> carpeta)
            model_name: Modelo de embeddings

        Returns:
            Diccionario de vectorstores cargados
        """
        loaded = {}

        for base_name in bases.keys():
            try:
                vs = self.load_vectorstore(base_name, model_name, auto_migrate=True)
                if vs:
                    loaded[base_name] = vs
            except Exception as e:
                log.error(f"No se pudo cargar {base_name}: {e}")

        log.info(f"✅ Cargadas {len(loaded)}/{len(bases)} bases")
        return loaded


# ======================
# Instancia Global
# ======================

VALIDATOR = EmbeddingValidator()


if __name__ == "__main__":
    # Test básico
    print("🔍 Embedding Validator - Test")
    print(f"Modelo por defecto: {EMBEDDING_MODEL} ({EMBEDDING_DIMENSION}d)")

    # Verificar dimensiones
    for model, dim in SUPPORTED_MODELS.items():
        print(f"  • {model}: {dim}d")
