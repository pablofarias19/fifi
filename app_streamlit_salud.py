# -*- coding: utf-8 -*-
"""
INTERFAZ STREAMLIT - Analizador Jurídico-Sanitario (Gemini Pro + RAG)
----------------------------------------------------------------------
Permite:
✅ Analizar texto o PDF médico-jurídico.
✅ Visualizar resultados JSON.
✅ Ver auditorías vectoriales.
✅ Generar informe PDF.
✅ Guardar y revisar historial de consultas.
✅ Sistema de versionado integrado.
"""

import streamlit as st
from pathlib import Path
import json, os, tempfile
from datetime import datetime

# Imports del sistema refactorizado
from config import BASES_RAG, DEFAULT_LLM
from analyser_salud import (
    make_llm, load_all_retrievers, analyse_text_medico,
    analyse_pdf_medico, render_pdf_report, audit_vectorstore,
    analyse_deep_layer
)
from version_manager import REGISTRY

# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(page_title="Analizador Jurídico-Sanitario", layout="wide")
st.title("🧠 Analizador Jurídico-Sanitario (Gemini Pro + RAG)")

HISTORIAL_PATH = Path("historial_salud.json")

# ============================================================
# SESSION STATE
# ============================================================

if "analisis_resultado" not in st.session_state:
    st.session_state["analisis_resultado"] = None
if "texto_analisis" not in st.session_state:
    st.session_state["texto_analisis"] = ""
if "llm" not in st.session_state:
    st.session_state["llm"] = None
if "retrievers" not in st.session_state:
    st.session_state["retrievers"] = None

# ============================================================
# VALIDACIÓN DE API KEY
# ============================================================

if "GOOGLE_API_KEY" not in os.environ or not os.environ["GOOGLE_API_KEY"].strip():
    st.warning("⚠️ Falta configurar GOOGLE_API_KEY en el entorno.")
    st.info("Configura la variable de entorno GOOGLE_API_KEY antes de usar la aplicación.")
    st.stop()
else:
    st.success("🔐 Clave de API de Gemini detectada.")

# ============================================================
# INICIALIZACIÓN (con caché)
# ============================================================

@st.cache_resource
def init_system():
    """Inicializa LLM y retrievers (cached)"""
    try:
        llm = make_llm(model=DEFAULT_LLM)
        retrievers = load_all_retrievers(BASES_RAG)
        return llm, retrievers
    except Exception as e:
        st.error(f"Error al inicializar sistema: {e}")
        return None, None

llm, retrievers = init_system()

if not retrievers:
    st.error("No se encontraron bases vectoriales en 'chroma_db_legal/*'.")
    st.info("Ejecuta la ingesta de documentos primero.")
    st.stop()
else:
    # Mostrar información de versiones
    with st.expander("📚 Bases disponibles y versiones"):
        for base_name in BASES_RAG.keys():
            active_version = REGISTRY.get_active_version(base_name)
            if active_version:
                meta = REGISTRY.get_version(base_name, active_version)
                st.write(f"**{base_name}**: v{active_version} ({meta.embedding_model}, {meta.total_fragments} fragmentos)")
            else:
                st.write(f"**{base_name}**: Sin versión activa")

# ============================================================
# MODO DE ANÁLISIS
# ============================================================

modo = st.radio("Seleccioná el modo de análisis:", ["Consulta", "Archivo PDF"])

result = {}
texto = ""

if modo == "Consulta":
    query = st.text_area("📝 Escribí tu consulta o caso médico:", height=200)
    if st.button("🔍 Analizar texto"):
        with st.spinner("Analizando..."):
            try:
                result = analyse_text_medico(query, retrievers, llm)
                texto = query
                st.session_state["analisis_resultado"] = result
                st.session_state["texto_analisis"] = texto
            except Exception as e:
                st.error(f"Error en análisis: {e}")

elif modo == "Archivo PDF":
    pdf_file = st.file_uploader("📄 Subí un archivo PDF", type=["pdf"])
    if pdf_file and st.button("🔍 Analizar PDF"):
        # Usar archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.read())
            tmp_path = Path(tmp.name)

        with st.spinner("Extrayendo texto y analizando..."):
            try:
                result = analyse_pdf_medico(tmp_path, retrievers, llm)
                texto = f"PDF subido: {pdf_file.name}"
                st.session_state["analisis_resultado"] = result
                st.session_state["texto_analisis"] = texto
            except Exception as e:
                st.error(f"Error en análisis: {e}")
            finally:
                # Limpiar archivo temporal
                tmp_path.unlink(missing_ok=True)

# Recuperar de session state si existe
if st.session_state["analisis_resultado"]:
    result = st.session_state["analisis_resultado"]
    texto = st.session_state["texto_analisis"]

# ============================================================
# MOSTRAR RESULTADOS
# ============================================================

if result:
    st.markdown("---")
    st.subheader("📋 Resumen estructurado")

    # Resumen en columnas
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Probabilidad de éxito", result.get("probabilidad_exito", "media").upper())

    with col2:
        fuentes = result.get("fuentes_relevantes", [])
        st.metric("Fuentes citadas", len(fuentes))

    # Tesis
    st.markdown("### 🎯 Tesis")
    st.info(result.get("tesis", "(sin tesis)"))

    # Conceptos clave
    conceptos = result.get("conceptos_clave", [])
    if conceptos:
        st.markdown("### 🔑 Conceptos clave")
        for concepto in conceptos:
            st.markdown(f"- {concepto}")

    # Debilidades
    debilidades = result.get("debilidades", [])
    if debilidades:
        st.markdown("### ⚠️ Debilidades")
        for deb in debilidades:
            st.markdown(f"- {deb}")

    # Preguntas derivadas
    preguntas = result.get("preguntas", [])
    if preguntas:
        st.markdown("### ❓ Preguntas derivadas")
        for preg in preguntas:
            st.markdown(f"- {preg}")

    # Informe narrativo completo
    if "texto_completo" in result and result["texto_completo"].strip():
        with st.expander("📄 Ver análisis completo de Gemini"):
            st.markdown(result["texto_completo"])

    # Fuentes relevantes
    if fuentes:
        with st.expander("🔗 Fuentes relevantes detectadas"):
            for i, f in enumerate(fuentes, 1):
                st.markdown(f"**{i}. {f.get('autor', '?')}** – *{f.get('titulo', '?')}* ({f.get('anio', 's/f')})")
                if f.get('pagina'):
                    st.caption(f"Página: {f['pagina']}")
                if f.get('url'):
                    st.caption(f"URL: {f['url']}")
                if f.get('tipo'):
                    st.caption(f"Tipo: {f['tipo']}")
                st.markdown("---")

    # ============================================================
    # AUDITORÍAS VECTORIALES
    # ============================================================

    st.markdown("---")
    st.subheader("🧮 Auditoría de bases vectoriales utilizadas")

    auditorias = []
    for nombre, vs in retrievers.items():
        audit = audit_vectorstore(vs, nombre)
        auditorias.append(audit)

    # Tabla de auditorías
    audit_data = []
    for a in auditorias:
        audit_data.append({
            "Base": a.base,
            "Fragmentos": a.frags,
            "Palabras prom.": f"{a.avg_words:.1f}",
            "Diversidad": f"{a.diversity:.2%}",
            "Cobertura": f"{a.coverage_types:.2%}",
            "Rating": f"{a.rating:.2f}/5.0"
        })

    st.table(audit_data)

    # ============================================================
    # GENERAR PDF
    # ============================================================

    st.markdown("---")
    if st.button("📥 Generar reporte PDF"):
        output_path = Path(tempfile.gettempdir()) / f"reporte_sanitario_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        titulo = f"Informe Jurídico-Sanitario (Nivel {result.get('nivel', 1)})"

        try:
            render_pdf_report(output_path, titulo, texto, result, auditorias)

            if output_path.exists():
                with open(output_path, "rb") as f:
                    st.download_button(
                        "⬇️ Descargar PDF",
                        f,
                        file_name=output_path.name,
                        mime="application/pdf"
                    )
                st.success("✅ PDF generado exitosamente")
            else:
                st.error("❌ Error: El archivo PDF no se creó")

        except Exception as e:
            st.error(f"❌ Error al generar PDF: {e}")

    # ============================================================
    # GUARDAR EN HISTORIAL
    # ============================================================

    if st.button("💾 Guardar en historial"):
        try:
            registro = {
                "timestamp": datetime.now().isoformat(),
                "modo": modo,
                "entrada": texto[:400],
                "resultado": result,
                "auditorias": [a.__dict__ for a in auditorias],
            }

            if HISTORIAL_PATH.exists():
                data = json.loads(HISTORIAL_PATH.read_text(encoding="utf-8"))
            else:
                data = []

            data.append(registro)
            HISTORIAL_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            st.success("✅ Consulta guardada en historial")

        except Exception as e:
            st.error(f"❌ Error guardando historial: {e}")

    # ============================================================
    # ANÁLISIS AVANZADO / ITERATIVO
    # ============================================================

    st.markdown("---")
    st.subheader("🔎 Análisis Avanzado / Iterativo")

    pregunta = st.text_area("Escribí una nueva consulta o instrucción para profundizar:", height=150)

    if st.button("🧠 Generar análisis avanzado"):
        with st.spinner("Profundizando el análisis jurídico..."):
            try:
                nivel_actual = int(result.get("nivel", 1)) + 1
                result_deep = analyse_deep_layer(result, llm, pregunta, nivel=nivel_actual)
                result_deep["nivel"] = nivel_actual

                st.success(f"✅ Análisis avanzado generado (Nivel {nivel_actual})")
                st.json(result_deep)

                # Guardar automáticamente
                registro = {
                    "timestamp": datetime.now().isoformat(),
                    "modo": "Iterativo",
                    "nivel": nivel_actual,
                    "entrada": pregunta[:400],
                    "resultado": result_deep,
                }

                if HISTORIAL_PATH.exists():
                    data = json.loads(HISTORIAL_PATH.read_text(encoding="utf-8"))
                else:
                    data = []

                data.append(registro)
                HISTORIAL_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

            except Exception as e:
                st.error(f"❌ Error en análisis avanzado: {e}")

# ============================================================
# PANEL DE HISTORIAL
# ============================================================

st.markdown("---")
st.subheader("🗂️ Historial de análisis anteriores")

if HISTORIAL_PATH.exists():
    try:
        data = json.loads(HISTORIAL_PATH.read_text(encoding="utf-8"))

        if data:
            opciones = [
                f"{i+1}. {d['timestamp']} - {d['modo']} (Nivel {d.get('nivel', 1)})"
                for i, d in enumerate(data)
            ]

            seleccion = st.selectbox("Seleccioná un análisis previo:", opciones)
            idx = opciones.index(seleccion)
            anterior = data[idx]

            col1, col2 = st.columns(2)

            with col1:
                if st.button("📖 Ver detalle"):
                    st.json(anterior["resultado"])

            with col2:
                if st.button("📤 Exportar a PDF"):
                    output_path = Path(tempfile.gettempdir()) / f"reporte_historial_{idx+1}.pdf"
                    try:
                        render_pdf_report(
                            output_path,
                            "Informe Sanitario (Historial)",
                            anterior["entrada"],
                            anterior["resultado"],
                            []
                        )

                        with open(output_path, "rb") as f:
                            st.download_button(
                                "⬇️ Descargar PDF",
                                f,
                                file_name=output_path.name,
                                mime="application/pdf"
                            )

                    except Exception as e:
                        st.error(f"Error exportando: {e}")

        else:
            st.info("📭 No hay análisis guardados aún.")

    except Exception as e:
        st.error(f"Error cargando historial: {e}")
else:
    st.info("📭 Aún no existe historial.")

# ============================================================
# FOOTER CON INFO DEL SISTEMA
# ============================================================

st.markdown("---")
with st.expander("ℹ️ Información del sistema"):
    stats = REGISTRY.get_stats()
    st.write(f"**Bases registradas:** {stats['total_bases']}")
    st.write(f"**Versiones totales:** {stats['total_versions']}")
    st.write(f"**Modelos en uso:** {', '.join(stats['models_in_use']) if stats['models_in_use'] else 'Ninguno'}")
    st.write(f"**LLM:** {DEFAULT_LLM}")
