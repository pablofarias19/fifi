[Contexto recuperado con metadatos y referencias]
{contexto}

[Texto o consulta principal]
{texto}

[INSTRUCCIONES DE ANÁLISIS]
1️⃣ **Síntesis contextual**
   - Resume el caso y su contexto jurídico-médico.
   - Determina tipo de responsabilidad (civil, penal, laboral, bioética).
   - Reconoce las fuentes primarias y secundarias relevantes.

2️⃣ **Ejes conceptuales**
   - Desarrolla los temas: consentimiento informado, lex artis, causalidad, daño, prueba.
   - Para cada eje: explica el concepto, cita fuentes con autor/año/página o tribunal/año.
   - Usa las referencias documentales para sostener los razonamientos.

3️⃣ **Evaluación argumental**
   - Señala los argumentos más sólidos y las debilidades.
   - Indica contradicciones entre jurisprudencias o doctrinas.
   - Identifica lagunas normativas o probatorias.

4️⃣ **Fuentes y trazabilidad**
   - Crea una lista de fuentes relevantes con formato:
     "Autor – Título (Año), pág. X [URL o Jurisdicción]"
   - Indica si son doctrina, jurisprudencia o normativa.
   - Recomienda cuáles podrían ampliarse con consultas específicas o documentos adicionales.

5️⃣ **Preguntas derivadas**
   - Formula 3 a 5 preguntas que profundicen el análisis.
   - Indica qué información o tipo de documento sería necesario obtener para resolverlas.

6️⃣ **Conclusión jurídica**
   - Formula una tesis integrada, valorando la probabilidad de éxito (baja/media/alta).
   - Justifica con criterios jurídicos y médicos.

[FORMATO DE SALIDA]
Primero entrega un informe narrativo completo con secciones jerarquizadas:
🔹 Resumen contextual
🔹 Ejes conceptuales
🔹 Debilidades / Vacíos argumentales
🔹 Preguntas de profundización
🔹 Conclusión jurídica y fuentes

Luego, entrega un bloque JSON resumido con esta estructura:

{
  "tesis": "...",
  "conceptos_clave": ["..."],
  "debilidades": ["..."],
  "preguntas": ["..."],
  "probabilidad_exito": "baja|media|alta",
  "fuentes_relevantes": [
    {"autor": "...", "titulo": "...", "anio": "...", "pagina": "...", "url": "...", "tipo": "..."}
  ]
}
