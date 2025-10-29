# app_streamlit_salud.py
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
"""

import streamlit as st
from pathlib import Path
import json, os
from datetime import datetime
from analyser_salud import (
    _make_llm, _load_all_retrievers, analyse_text_medico,
    analyse_pdf_medico, render_pdf_report, _audit_vectorstore,
    BASES_RAG, DB_DIR_DEFAULT, DEFAULT_LLM
)

# ============================================================
# MANEJO DE SESSION STATE SEGURO
# ============================================================
if "uploaded_pdf" not in st.session_state:
    st.session_state["uploaded_pdf"] = None
if "analisis_resultado" not in st.session_state:
    st.session_state["analisis_resultado"] = None
if "texto_analisis" not in st.session_state:
    st.session_state["texto_analisis"] = ""

# ======================
# CONFIGURACIÓN BASE
# ======================

st.set_page_config(page_title="Analizador Jurídico-Sanitario", layout="wide")
st.title("🧠 Analizador Jurídico-Sanitario (Gemini Pro + RAG)")

HISTORIAL_PATH = Path("historial_salud.json")

# ======================
# CLAVE GEMINI
# ======================
if "GOOGLE_API_KEY" not in os.environ or not os.environ["GOOGLE_API_KEY"].strip():
    st.warning("⚠️ Falta configurar GOOGLE_API_KEY en el entorno.")
else:
    st.success("🔐 Clave de API de Gemini detectada.")

# ======================
# CARGA MODELO Y BASES
# ======================
try:
    llm = _make_llm(model=DEFAULT_LLM)
    retrievers = _load_all_retrievers(DB_DIR_DEFAULT, BASES_RAG)
    if not retrievers:
        st.error("No se encontraron bases vectoriales en 'chroma_db_legal/*'.")
    else:
        st.info(f"📚 Bases disponibles: {', '.join(retrievers.keys())}")
except Exception as e:
    st.error(f"Error al inicializar: {e}")
    st.stop()

# ======================
# MODO DE ANÁLISIS
# ======================
modo = st.radio("Seleccioná el modo de análisis:", ["Consulta", "Archivo PDF"])
texto = ""
result = {}
auditorias = []

if modo == "Consulta":
    query = st.text_area("📝 Escribí tu consulta o caso médico:", height=200)
    if st.button("🔍 Analizar texto"):
        with st.spinner("Analizando..."):
            result = analyse_text_medico(query, retrievers, llm)
            texto = query

elif modo == "Archivo PDF":
    pdf_file = st.file_uploader("📄 Subí un archivo PDF", type=["pdf"])
    if pdf_file and st.button("🔍 Analizar PDF"):
        tmp = Path("temp_input.pdf")
        tmp.write_bytes(pdf_file.read())
        with st.spinner("Extrayendo texto y analizando..."):
            result = analyse_pdf_medico(tmp, retrievers, llm)
            texto = f"PDF subido: {pdf_file.name}"

# ======================
# MOSTRAR RESULTADOS ENRIQUECIDOS
# ======================

if result:
    st.subheader("📋 Resumen estructurado (síntesis del modelo)")
    resumen = {
        "Tesis": result.get("tesis"),
        "Conceptos clave": result.get("conceptos_clave"),
        "Debilidades": result.get("debilidades"),
        "Preguntas derivadas": result.get("preguntas"),
    "Probabilidad de éxito": result.get("probabilidad_exito"),
    }
    st.json(resumen, expanded=False)

    # 🔍 Informe narrativo completo
    if "texto_completo" in result and result["texto_completo"].strip():
        st.markdown("### 🧠 Análisis completo de Gemini (con citas y trazabilidad)")
        st.markdown(result["texto_completo"])

    # 🔗 Fuentes relevantes reutilizables
    fuentes = result.get("fuentes_relevantes", [])
    if fuentes:
        st.markdown("### 🔗 Fuentes relevantes detectadas")
        st.table([
            {
                "Autor": f.get("autor", ""),
                "Título": f.get("titulo", ""),
                "Año": f.get("anio", ""),
                "Página": f.get("pagina", ""),
                "Tipo": f.get("tipo", ""),
                "URL": f.get("url", "")
            } for f in fuentes
        ])
        st.info("💡 Podés reutilizar estas referencias en consultas posteriores o análisis comparativos.")

    # Auditorías vectoriales
    st.markdown("---")
    st.subheader("🧮 Auditoría de bases vectoriales utilizadas")
    auditorias = [_audit_vectorstore(vs, nombre) for nombre, vs in retrievers.items()]
    st.dataframe(
        [{"Base": a.base, "Fragments": a.frags, "Avg Words": a.avg_words,
          "Diversidad": a.diversity, "Cobertura": a.coverage_types, "Rating": a.rating}
         for a in auditorias],
        use_container_width=True
    )

    # 📥 Generar informe PDF completo
    if st.button("📥 Generar reporte PDF con fuentes"):
        out_path = Path("reporte_sanitario_fuentes.pdf")
        titulo = f"Informe Jurídico-Sanitario con Fuentes (Nivel {result.get('nivel',1)})"
        st.info(f"[LOG] Creando PDF en: {out_path}")
        try:
            render_pdf_report(out_path, titulo, texto, result, auditorias)
            st.info(f"[LOG] PDF generado exitosamente: {out_path.exists()}")
            if out_path.exists():
                with open(out_path, "rb") as f:
                    st.download_button("⬇️ Descargar PDF", f, file_name="reporte_sanitario_fuentes.pdf")
                    st.success("[LOG] PDF listo para descargar.")
            else:
                st.error("[LOG] Error: El archivo PDF no se creó correctamente.")
        except Exception as e:
            st.error(f"[LOG] Error al generar o descargar PDF: {e}")

    # Guardar en historial
    if st.button("💾 Guardar en historial"):
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
        st.success("✅ Consulta guardada en historial.")

    # ======================
    # ANÁLISIS AVANZADO / ITERATIVO
    # ======================
    st.markdown("---")
    st.subheader("🔎 Análisis Avanzado / Iterativo")

    pregunta = st.text_area("Escribí una nueva consulta o instrucción para profundizar:", height=150)

    if st.button("🧠 Generar análisis avanzado"):
        with st.spinner("Profundizando el análisis jurídico..."):
            try:
                # Determinamos el nivel según si el resultado ya contiene un campo 'nivel'
                nivel_actual = int(result.get("nivel", 1)) + 1
                from analyser_salud import analyse_deep_layer
                result_deep = analyse_deep_layer(result, llm, pregunta, nivel=nivel_actual)
                result_deep["nivel"] = nivel_actual
                st.success(f"✅ Análisis avanzado generado (Nivel {nivel_actual})")
                st.json(result_deep, expanded=False)

                # Guardamos la iteración en historial extendido
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
                st.error(f"Error en análisis avanzado: {e}")

# ======================
# PANEL DE HISTORIAL
# ======================
st.markdown("---")
st.subheader("🗂️ Historial de análisis anteriores")

if HISTORIAL_PATH.exists():
    data = json.loads(HISTORIAL_PATH.read_text(encoding="utf-8"))
    if data:
        opciones = [f"{i+1}. {d['timestamp']} - {d['modo']} (Nivel {d.get('nivel',1)})" for i, d in enumerate(data)]
        seleccion = st.selectbox("Seleccioná un análisis previo:", opciones)
        idx = opciones.index(seleccion)
        anterior = data[idx]
        if st.button("📖 Ver detalle del análisis seleccionado"):
            st.json(anterior["resultado"])
        if st.button("📤 Exportar análisis a PDF"):
            out_path = Path(f"reporte_{idx+1}.pdf")
            render_pdf_report(out_path, "Informe Sanitario (Historial)", anterior["entrada"],
                              anterior["resultado"], [])
            with open(out_path, "rb") as f:
                st.download_button("⬇️ Descargar PDF", f, file_name=out_path.name)
    else:
        st.info("📭 No hay análisis guardados aún.")
else:
    st.info("📭 Aún no existe historial.")
