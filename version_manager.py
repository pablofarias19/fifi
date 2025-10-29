# -*- coding: utf-8 -*-
"""
VERSION MANAGER - Sistema de Versionado de Bases Vectoriales
-------------------------------------------------------------
Gestiona versiones, migraciones y compatibilidad de embeddings.
"""

from __future__ import annotations
import json
import shutil
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

from config import (
    DB_DIR_DEFAULT, REGISTRY_PATH, EMBEDDING_MODEL,
    EMBEDDING_DIMENSION, BASES_RAG, SUPPORTED_MODELS
)

# ======================
# Logger
# ======================

log = logging.getLogger("version_manager")
log.setLevel(logging.INFO)
if not log.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)

# ======================
# Modelos de Datos
# ======================

@dataclass
class BaseVersion:
    """Metadata de una versión específica de base vectorial"""
    version: str                    # "1.0.0"
    embedding_model: str            # "sentence-transformers/all-mpnet-base-v2"
    embedding_dim: int              # 768
    total_docs: int                 # 1234
    total_fragments: int            # 5678
    created_at: str                 # ISO timestamp
    last_updated: str               # ISO timestamp
    migration_from: Optional[str]   # "0.9.0" (si migró desde otra versión)
    quality_score: float            # 0.0 - 5.0 (de auditoría)
    is_active: bool                 # True si es la versión en uso

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MigrationPlan:
    """Plan de migración entre versiones"""
    base_name: str
    from_version: str
    to_version: str
    from_model: str
    to_model: str
    docs_to_migrate: int
    estimated_time_minutes: float
    requires_reingestion: bool

    def to_dict(self) -> dict:
        return asdict(self)


# ======================
# Registry - Gestión de Versiones
# ======================

class VersionRegistry:
    """
    Registro centralizado de todas las versiones de bases vectoriales.
    Almacena metadata y gestiona versiones activas.
    """

    def __init__(self, registry_path: Path = REGISTRY_PATH):
        self.registry_path = registry_path
        self.data: Dict[str, Dict[str, dict]] = self._load()

    def _load(self) -> Dict[str, Dict[str, dict]]:
        """Carga registro desde JSON"""
        if not self.registry_path.exists():
            log.info(f"Creando nuevo registro en {self.registry_path}")
            return {}

        try:
            data = json.loads(self.registry_path.read_text(encoding="utf-8"))
            log.info(f"Registro cargado: {len(data)} bases registradas")
            return data
        except Exception as e:
            log.error(f"Error cargando registro: {e}")
            return {}

    def _save(self) -> None:
        """Persiste registro a JSON"""
        try:
            self.registry_path.write_text(
                json.dumps(self.data, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            log.debug("Registro guardado exitosamente")
        except Exception as e:
            log.error(f"Error guardando registro: {e}")

    # ======================
    # Operaciones CRUD
    # ======================

    def register_version(self, base_name: str, version: BaseVersion) -> None:
        """Registra nueva versión de una base"""
        if base_name not in self.data:
            self.data[base_name] = {}

        # Desactivar versión anterior si esta es activa
        if version.is_active:
            for v in self.data[base_name].values():
                v["is_active"] = False

        self.data[base_name][version.version] = version.to_dict()
        self._save()
        log.info(f"✅ Versión {version.version} registrada para {base_name}")

    def get_version(self, base_name: str, version: str) -> Optional[BaseVersion]:
        """Obtiene metadata de versión específica"""
        if base_name not in self.data or version not in self.data[base_name]:
            return None

        data = self.data[base_name][version]
        return BaseVersion(**data)

    def get_active_version(self, base_name: str) -> Optional[str]:
        """Obtiene versión activa de una base"""
        if base_name not in self.data:
            return None

        for version, meta in self.data[base_name].items():
            if meta.get("is_active", False):
                return version

        # Si no hay activa, devolver la más reciente
        versions = list(self.data[base_name].keys())
        if versions:
            return self._get_latest_version(versions)

        return None

    def set_active_version(self, base_name: str, version: str) -> bool:
        """Marca una versión como activa"""
        if base_name not in self.data or version not in self.data[base_name]:
            log.error(f"Versión {version} no existe para {base_name}")
            return False

        # Desactivar todas las versiones
        for v in self.data[base_name].values():
            v["is_active"] = False

        # Activar la solicitada
        self.data[base_name][version]["is_active"] = True
        self._save()
        log.info(f"✅ Versión {version} activada para {base_name}")
        return True

    def list_versions(self, base_name: str) -> List[str]:
        """Lista todas las versiones de una base"""
        if base_name not in self.data:
            return []
        return sorted(self.data[base_name].keys(), key=self._version_sort_key, reverse=True)

    def list_all_bases(self) -> List[str]:
        """Lista todas las bases registradas"""
        return list(self.data.keys())

    # ======================
    # Operaciones de Migración
    # ======================

    def create_migration_plan(
        self,
        base_name: str,
        target_model: str
    ) -> Optional[MigrationPlan]:
        """Genera plan de migración automático"""
        active_version = self.get_active_version(base_name)
        if not active_version:
            log.error(f"No hay versión activa para {base_name}")
            return None

        current_meta = self.data[base_name][active_version]

        # Calcular nueva versión (incremento major)
        new_version = self._bump_version(active_version, "major")

        # Estimar tiempo (aprox 0.5 seg por documento)
        docs = current_meta.get("total_docs", 0)
        estimated_time = (docs * 0.5) / 60.0  # minutos

        # Determinar si requiere re-ingesta (si cambia dimensión)
        current_dim = current_meta.get("embedding_dim", 0)
        target_dim = SUPPORTED_MODELS.get(target_model, 0)
        requires_reingestion = (current_dim != target_dim)

        return MigrationPlan(
            base_name=base_name,
            from_version=active_version,
            to_version=new_version,
            from_model=current_meta["embedding_model"],
            to_model=target_model,
            docs_to_migrate=docs,
            estimated_time_minutes=estimated_time,
            requires_reingestion=requires_reingestion
        )

    def get_compatible_version(
        self,
        base_name: str,
        model: str
    ) -> Optional[str]:
        """Busca versión compatible con un modelo específico"""
        if base_name not in self.data:
            return None

        for version, meta in self.data[base_name].items():
            if meta["embedding_model"] == model:
                return version

        return None

    # ======================
    # Utilidades
    # ======================

    def _get_latest_version(self, versions: List[str]) -> str:
        """Obtiene la versión más reciente (semver)"""
        return sorted(versions, key=self._version_sort_key, reverse=True)[0]

    def _version_sort_key(self, version: str) -> Tuple[int, int, int]:
        """Convierte versión string a tupla para ordenar"""
        try:
            parts = version.split(".")
            return (int(parts[0]), int(parts[1]), int(parts[2]))
        except (ValueError, IndexError):
            return (0, 0, 0)

    def _bump_version(self, version: str, level: str = "minor") -> str:
        """Incrementa versión semántica"""
        try:
            major, minor, patch = map(int, version.split("."))

            if level == "major":
                return f"{major + 1}.0.0"
            elif level == "minor":
                return f"{major}.{minor + 1}.0"
            elif level == "patch":
                return f"{major}.{minor}.{patch + 1}"
            else:
                return version
        except ValueError:
            return "1.0.0"

    def get_stats(self) -> dict:
        """Estadísticas del registro"""
        total_bases = len(self.data)
        total_versions = sum(len(versions) for versions in self.data.values())

        models_used = set()
        for base_data in self.data.values():
            for version_data in base_data.values():
                models_used.add(version_data["embedding_model"])

        return {
            "total_bases": total_bases,
            "total_versions": total_versions,
            "models_in_use": list(models_used),
            "bases": list(self.data.keys())
        }


# ======================
# Funciones de Directorio
# ======================

def get_version_directory(base_name: str, version: str) -> Path:
    """Obtiene path del directorio de una versión específica"""
    if base_name not in BASES_RAG:
        raise ValueError(f"Base desconocida: {base_name}")

    base_folder = BASES_RAG[base_name]
    version_folder = f"v{version}"
    return DB_DIR_DEFAULT / base_folder / version_folder


def create_version_directory(base_name: str, version: str) -> Path:
    """Crea directorio para nueva versión"""
    version_dir = get_version_directory(base_name, version)
    version_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"📁 Directorio creado: {version_dir}")
    return version_dir


def backup_version(base_name: str, version: str) -> Optional[Path]:
    """Crea backup de una versión"""
    try:
        source_dir = get_version_directory(base_name, version)
        if not source_dir.exists():
            log.warning(f"Directorio no existe: {source_dir}")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = source_dir.parent / f"backup_{version}_{timestamp}"

        shutil.copytree(source_dir, backup_dir)
        log.info(f"💾 Backup creado: {backup_dir}")
        return backup_dir

    except Exception as e:
        log.error(f"Error creando backup: {e}")
        return None


# ======================
# Inicialización Global
# ======================

# Instancia global del registro
REGISTRY = VersionRegistry()


if __name__ == "__main__":
    # Test básico
    print("📋 Version Manager - Test")
    print(f"Registry path: {REGISTRY.registry_path}")
    print(f"Stats: {REGISTRY.get_stats()}")

    # Listar bases
    for base in REGISTRY.list_all_bases():
        active = REGISTRY.get_active_version(base)
        print(f"  • {base}: activa={active}")
