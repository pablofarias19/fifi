# -*- coding: utf-8 -*-
"""
AUDIT BASES - Herramienta de Auditoría de Bases Vectoriales
------------------------------------------------------------
Audita, inspecciona y gestiona versiones de bases vectoriales.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from config import BASES_RAG
from version_manager import REGISTRY, get_version_directory
from embedding_validator import VALIDATOR
from analyser_salud import audit_vectorstore


def show_registry_stats():
    """Muestra estadísticas del registro de versiones"""
    stats = REGISTRY.get_stats()

    print("\n📊 ESTADÍSTICAS DEL REGISTRO")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Bases registradas: {stats['total_bases']}")
    print(f"Versiones totales: {stats['total_versions']}")
    print(f"Modelos en uso: {', '.join(stats['models_in_use']) if stats['models_in_use'] else 'Ninguno'}")
    print(f"\nBases: {', '.join(stats['bases']) if stats['bases'] else 'Ninguna'}")


def list_all_bases():
    """Lista todas las bases con sus versiones"""
    bases = REGISTRY.list_all_bases()

    if not bases:
        print("\n📭 No hay bases registradas")
        return

    print("\n📚 BASES VECTORIALES")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    for base_name in sorted(bases):
        versions = REGISTRY.list_versions(base_name)
        active_version = REGISTRY.get_active_version(base_name)

        print(f"\n🗂️  {base_name}")
        if not versions:
            print(f"   └─ (sin versiones)")
            continue

        for version in versions:
            meta = REGISTRY.get_version(base_name, version)
            if not meta:
                continue

            # Indicador de activa
            active_mark = "✅ ACTIVA" if meta.is_active or version == active_version else ""

            # Info básica
            print(f"   └─ v{version} {active_mark}")
            print(f"      ├─ Modelo: {meta.embedding_model} ({meta.embedding_dim}d)")
            print(f"      ├─ Documentos: {meta.total_docs} ({meta.total_fragments} fragmentos)")
            print(f"      ├─ Calidad: {meta.quality_score:.2f}/5.0")
            print(f"      ├─ Creada: {meta.created_at}")

            if meta.migration_from:
                print(f"      └─ Migrada desde: v{meta.migration_from}")


def audit_base(base_name: str, version: Optional[str] = None):
    """
    Audita una base específica con métricas detalladas.

    Args:
        base_name: Nombre de la base
        version: Versión específica (None = activa)
    """
    if base_name not in BASES_RAG:
        print(f"❌ Error: Base '{base_name}' no existe")
        return

    # Determinar versión
    if not version:
        version = REGISTRY.get_active_version(base_name)
        if not version:
            print(f"❌ Error: No hay versión activa para '{base_name}'")
            return

    # Obtener metadata
    meta = REGISTRY.get_version(base_name, version)
    if not meta:
        print(f"❌ Error: Versión {version} no encontrada")
        return

    print(f"\n🔍 AUDITORÍA: {base_name} v{version}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # Metadata de versión
    print(f"\n📋 METADATA")
    print(f"   Modelo: {meta.embedding_model}")
    print(f"   Dimensión: {meta.embedding_dim}d")
    print(f"   Documentos: {meta.total_docs}")
    print(f"   Fragmentos: {meta.total_fragments}")
    print(f"   Calidad registrada: {meta.quality_score:.2f}/5.0")
    print(f"   Estado: {'🟢 ACTIVA' if meta.is_active else '⚪ INACTIVA'}")
    print(f"   Creada: {meta.created_at}")
    print(f"   Última actualización: {meta.last_updated}")
    if meta.migration_from:
        print(f"   Migrada desde: v{meta.migration_from}")

    # Verificar directorio
    version_dir = get_version_directory(base_name, version)
    if not version_dir.exists():
        print(f"\n⚠️  Advertencia: Directorio no existe: {version_dir}")
        return

    print(f"\n📂 Directorio: {version_dir}")

    # Cargar vectorstore
    print(f"\n🔄 Cargando vectorstore...")
    try:
        vs = VALIDATOR.load_vectorstore(base_name, meta.embedding_model)
        if not vs:
            print(f"❌ Error: No se pudo cargar vectorstore")
            return
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # Auditoría vectorial
    print(f"\n🧮 ANÁLISIS VECTORIAL (muestreo: 400 docs)")
    audit = audit_vectorstore(vs, base_name, sample=400)

    print(f"   Fragmentos muestreados: {audit.frags}")
    print(f"   Palabras promedio: {audit.avg_words:.1f}")
    print(f"   Diversidad léxica: {audit.diversity:.2%}")
    print(f"   Cobertura de tipos: {audit.coverage_types:.2%}")
    print(f"   Rating calculado: {audit.rating:.2f}/5.0")

    # Interpretación
    print(f"\n📊 INTERPRETACIÓN")
    if audit.rating >= 4.0:
        print(f"   ✅ Excelente calidad - Base bien estructurada")
    elif audit.rating >= 3.0:
        print(f"   ✔️  Buena calidad - Aceptable para producción")
    elif audit.rating >= 2.0:
        print(f"   ⚠️  Calidad media - Considerar re-ingesta")
    else:
        print(f"   ❌ Baja calidad - Requiere re-ingesta")

    if audit.avg_words < 50:
        print(f"   ⚠️  Fragmentos muy cortos (< 50 palabras)")
    elif audit.avg_words > 1000:
        print(f"   ⚠️  Fragmentos muy largos (> 1000 palabras)")

    if audit.diversity < 0.3:
        print(f"   ⚠️  Baja diversidad léxica (mucha repetición)")

    if audit.coverage_types < 0.4:
        print(f"   ⚠️  Baja cobertura de tipos estructurales")


def activate_version(base_name: str, version: str):
    """
    Activa una versión específica de una base.

    Args:
        base_name: Nombre de la base
        version: Versión a activar
    """
    if base_name not in BASES_RAG:
        print(f"❌ Error: Base '{base_name}' no existe")
        return False

    # Verificar que la versión existe
    meta = REGISTRY.get_version(base_name, version)
    if not meta:
        print(f"❌ Error: Versión {version} no encontrada")
        return False

    # Activar
    if REGISTRY.set_active_version(base_name, version):
        print(f"✅ Versión {version} activada para '{base_name}'")
        return True
    else:
        print(f"❌ Error activando versión")
        return False


def compare_versions(base_name: str, version1: str, version2: str):
    """
    Compara dos versiones de una base.

    Args:
        base_name: Nombre de la base
        version1: Primera versión
        version2: Segunda versión
    """
    if base_name not in BASES_RAG:
        print(f"❌ Error: Base '{base_name}' no existe")
        return

    # Obtener metadatas
    meta1 = REGISTRY.get_version(base_name, version1)
    meta2 = REGISTRY.get_version(base_name, version2)

    if not meta1:
        print(f"❌ Error: Versión {version1} no encontrada")
        return
    if not meta2:
        print(f"❌ Error: Versión {version2} no encontrada")
        return

    print(f"\n⚖️  COMPARACIÓN: {base_name}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # Tabla comparativa
    print(f"\n{'Métrica':<25} | {'v' + version1:<20} | {'v' + version2:<20}")
    print("─" * 70)
    print(f"{'Modelo':<25} | {meta1.embedding_model[:18]:<20} | {meta2.embedding_model[:18]:<20}")
    print(f"{'Dimensión':<25} | {str(meta1.embedding_dim) + 'd':<20} | {str(meta2.embedding_dim) + 'd':<20}")
    print(f"{'Documentos':<25} | {meta1.total_docs:<20} | {meta2.total_docs:<20}")
    print(f"{'Fragmentos':<25} | {meta1.total_fragments:<20} | {meta2.total_fragments:<20}")
    print(f"{'Calidad':<25} | {f'{meta1.quality_score:.2f}/5.0':<20} | {f'{meta2.quality_score:.2f}/5.0':<20}")
    print(f"{'Estado':<25} | {'ACTIVA' if meta1.is_active else 'INACTIVA':<20} | {'ACTIVA' if meta2.is_active else 'INACTIVA':<20}")

    # Recomendación
    print(f"\n💡 RECOMENDACIÓN")
    if meta1.quality_score > meta2.quality_score:
        print(f"   v{version1} tiene mejor calidad ({meta1.quality_score:.2f} vs {meta2.quality_score:.2f})")
    elif meta2.quality_score > meta1.quality_score:
        print(f"   v{version2} tiene mejor calidad ({meta2.quality_score:.2f} vs {meta1.quality_score:.2f})")
    else:
        print(f"   Ambas versiones tienen calidad similar")


def main():
    parser = argparse.ArgumentParser(
        description="Audita y gestiona bases vectoriales",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Mostrar todas las bases y versiones
  python audit_bases.py --list

  # Auditar base activa
  python audit_bases.py --audit "Salud - Médica"

  # Auditar versión específica
  python audit_bases.py --audit "Salud - Médica" --version "1.0.0"

  # Activar versión
  python audit_bases.py --activate "Salud - Médica" "2.0.0"

  # Comparar versiones
  python audit_bases.py --compare "Salud - Médica" "1.0.0" "2.0.0"

  # Estadísticas del registro
  python audit_bases.py --stats
        """
    )

    parser.add_argument("--list", action="store_true", help="Listar todas las bases y versiones")
    parser.add_argument("--audit", metavar="BASE", help="Auditar una base específica")
    parser.add_argument("--version", metavar="VER", help="Versión específica a auditar")
    parser.add_argument("--activate", nargs=2, metavar=("BASE", "VERSION"), help="Activar versión")
    parser.add_argument("--compare", nargs=3, metavar=("BASE", "V1", "V2"), help="Comparar dos versiones")
    parser.add_argument("--stats", action="store_true", help="Mostrar estadísticas del registro")

    args = parser.parse_args()

    # Sin argumentos, mostrar ayuda
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    # Ejecutar comando
    if args.stats:
        show_registry_stats()

    if args.list:
        list_all_bases()

    if args.audit:
        audit_base(args.audit, args.version)

    if args.activate:
        activate_version(args.activate[0], args.activate[1])

    if args.compare:
        compare_versions(args.compare[0], args.compare[1], args.compare[2])


if __name__ == "__main__":
    main()
