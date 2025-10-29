# -*- coding: utf-8 -*-
"""
MIGRATE CLI - Herramienta de Migración de Bases Vectoriales
------------------------------------------------------------
Migra bases vectoriales a nuevos modelos de embeddings.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from config import BASES_RAG, SUPPORTED_MODELS
from version_manager import (
    REGISTRY, BaseVersion, create_version_directory,
    get_version_directory, backup_version
)
from embedding_validator import VALIDATOR
from langchain_chroma import Chroma
from datetime import datetime

def migrate_base(
    base_name: str,
    new_model: str,
    batch_size: int = 100,
    create_backup: bool = True
) -> bool:
    """
    Migra una base vectorial a un nuevo modelo de embeddings.

    Args:
        base_name: Nombre de la base a migrar
        new_model: Nuevo modelo de embeddings
        batch_size: Tamaño de lote para procesamiento
        create_backup: Si crear backup antes de migrar

    Returns:
        True si migración exitosa, False si falla
    """
    print(f"\n🔄 MIGRACIÓN DE BASE VECTORIAL")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Base: {base_name}")
    print(f"Modelo destino: {new_model}")

    # Validar base
    if base_name not in BASES_RAG:
        print(f"❌ Error: Base '{base_name}' no existe")
        return False

    # Validar modelo
    if new_model not in SUPPORTED_MODELS:
        print(f"⚠️  Advertencia: Modelo '{new_model}' no está en modelos soportados")
        print(f"   Modelos conocidos: {list(SUPPORTED_MODELS.keys())}")
        respuesta = input("¿Continuar de todos modos? (s/N): ")
        if respuesta.lower() != 's':
            return False

    # Obtener versión activa
    active_version = REGISTRY.get_active_version(base_name)
    if not active_version:
        print(f"❌ Error: No hay versión activa para '{base_name}'")
        return False

    print(f"\n📊 Versión actual: {active_version}")

    # Cargar metadata de versión actual
    current_meta = REGISTRY.get_version(base_name, active_version)
    if not current_meta:
        print(f"❌ Error: No se pudo cargar metadata de versión {active_version}")
        return False

    print(f"   Modelo actual: {current_meta.embedding_model} ({current_meta.embedding_dim}d)")
    print(f"   Documentos: {current_meta.total_docs}")
    print(f"   Fragmentos: {current_meta.total_fragments}")

    # Verificar si ya usa el modelo destino
    if current_meta.embedding_model == new_model:
        print(f"\n✅ La base ya usa el modelo {new_model}")
        return True

    # Crear plan de migración
    plan = REGISTRY.create_migration_plan(base_name, new_model)
    if not plan:
        print(f"❌ Error: No se pudo crear plan de migración")
        return False

    print(f"\n📋 PLAN DE MIGRACIÓN")
    print(f"   De: {plan.from_model} ({current_meta.embedding_dim}d)")
    print(f"   A:  {plan.to_model} ({SUPPORTED_MODELS.get(new_model, '?')}d)")
    print(f"   Documentos a migrar: {plan.docs_to_migrate}")
    print(f"   Tiempo estimado: {plan.estimated_time_minutes:.1f} minutos")
    print(f"   Requiere re-ingesta: {'Sí' if plan.requires_reingestion else 'No'}")

    # Confirmar
    respuesta = input("\n¿Proceder con la migración? (s/N): ")
    if respuesta.lower() != 's':
        print("❌ Migración cancelada")
        return False

    # Backup (opcional)
    if create_backup:
        print(f"\n💾 Creando backup de versión {active_version}...")
        backup_path = backup_version(base_name, active_version)
        if backup_path:
            print(f"   ✅ Backup creado: {backup_path}")
        else:
            print(f"   ⚠️  No se pudo crear backup (continuando...)")

    # Cargar base actual
    print(f"\n📂 Cargando base actual...")
    try:
        old_vs = VALIDATOR.load_vectorstore(base_name, current_meta.embedding_model)
        if not old_vs:
            print(f"❌ Error: No se pudo cargar vectorstore")
            return False
    except Exception as e:
        print(f"❌ Error cargando vectorstore: {e}")
        return False

    # Obtener todos los documentos
    print(f"📥 Extrayendo documentos...")
    try:
        # Chroma no tiene método get_all, así que hacemos un similarity_search grande
        all_docs = old_vs.similarity_search("", k=100000)  # Obtener todos
        print(f"   ✅ {len(all_docs)} documentos extraídos")
    except Exception as e:
        print(f"❌ Error extrayendo documentos: {e}")
        return False

    # Crear nueva versión
    new_version = plan.to_version
    print(f"\n🆕 Creando nueva versión: {new_version}")
    version_dir = create_version_directory(base_name, new_version)

    # Cargar nuevo modelo de embeddings
    print(f"🧠 Cargando modelo {new_model}...")
    try:
        new_embeddings = VALIDATOR.get_embeddings(new_model)
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return False

    # Crear nuevo vectorstore
    new_vs = Chroma(
        collection_name="legal_fragments",
        embedding_function=new_embeddings,
        persist_directory=str(version_dir)
    )

    # Re-ingestar documentos con progress bar
    print(f"\n⚙️  Re-ingesta de documentos (batch_size={batch_size})...")
    failed = 0
    with tqdm(total=len(all_docs), desc="Migrando", unit="doc") as pbar:
        for i in range(0, len(all_docs), batch_size):
            batch = all_docs[i:i+batch_size]
            try:
                new_vs.add_documents(batch)
                pbar.update(len(batch))
            except Exception as e:
                print(f"\n⚠️  Error en batch {i//batch_size + 1}: {e}")
                failed += len(batch)
                pbar.update(len(batch))

    if failed > 0:
        print(f"\n⚠️  {failed} documentos fallaron durante la migración")

    # Registrar nueva versión
    new_meta = BaseVersion(
        version=new_version,
        embedding_model=new_model,
        embedding_dim=SUPPORTED_MODELS.get(new_model, 0),
        total_docs=current_meta.total_docs,
        total_fragments=len(all_docs) - failed,
        created_at=datetime.now().isoformat(),
        last_updated=datetime.now().isoformat(),
        migration_from=active_version,
        quality_score=current_meta.quality_score,
        is_active=False  # No activar automáticamente
    )

    REGISTRY.register_version(base_name, new_meta)
    print(f"\n✅ Nueva versión registrada: {new_version}")

    # Preguntar si activar
    respuesta = input(f"\n¿Activar versión {new_version} como versión activa? (s/N): ")
    if respuesta.lower() == 's':
        REGISTRY.set_active_version(base_name, new_version)
        print(f"✅ Versión {new_version} activada")
    else:
        print(f"ℹ️  Versión {new_version} creada pero no activada")
        print(f"   Para activarla: python audit_bases.py --activate {base_name} {new_version}")

    print(f"\n🎉 MIGRACIÓN COMPLETADA")
    print(f"   Base: {base_name}")
    print(f"   Versión: {active_version} → {new_version}")
    print(f"   Documentos: {len(all_docs) - failed}/{len(all_docs)}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Migra bases vectoriales a nuevos modelos de embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Migrar Salud Médica a mpnet-768d
  python migrate_cli.py --base "Salud - Médica" --model sentence-transformers/all-mpnet-base-v2

  # Migrar sin backup
  python migrate_cli.py --base "Salud - Laboral" --model sentence-transformers/all-mpnet-base-v2 --no-backup

  # Migrar con batch size mayor
  python migrate_cli.py --base "Jurisprudencia - Salud" --model sentence-transformers/all-mpnet-base-v2 --batch-size 200
        """
    )

    parser.add_argument(
        "--base",
        required=True,
        choices=list(BASES_RAG.keys()),
        help="Base a migrar"
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Modelo de embeddings destino"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Tamaño de lote para procesamiento (default: 100)"
    )

    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="No crear backup antes de migrar"
    )

    args = parser.parse_args()

    # Ejecutar migración
    success = migrate_base(
        base_name=args.base,
        new_model=args.model,
        batch_size=args.batch_size,
        create_backup=not args.no_backup
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
