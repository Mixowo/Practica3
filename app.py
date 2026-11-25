import joblib
import pandas as pd
import streamlit as st
from groq import Groq

st.set_page_config(page_title="Mantenimiento Predictivo", page_icon="🏭", layout="wide")

ESTADOS = {
    0: "Operación Normal",
    1: "Mantenimiento Preventivo",
    2: "Fallo Crítico",
}


# ---------- Modelo (solo depende del .pkl) ----------
@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_mantenimiento.pkl")


modelo = cargar_modelo()

# ---------- Controles laterales ----------
st.sidebar.header("🎛️ Lecturas de sensores")
temperatura = st.sidebar.slider("Temperatura (°C)", 50.0, 110.0, 60.0, 0.5)
vibracion = st.sidebar.slider("Vibración (mm/s)", 0.0, 12.0, 1.5, 0.1)
horas = st.sidebar.slider("Horas de operación", 0, 5000, 300, 10)
presion = st.sidebar.slider("Presión (PSI)", 80.0, 160.0, 92.0, 0.5)

st.title("🏭 Sistema Híbrido de Mantenimiento Predictivo")

# ---------- Métricas ----------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Temperatura", f"{temperatura} °C")
col2.metric("Vibración", f"{vibracion} mm/s")
col3.metric("Operación", f"{horas} hrs")
col4.metric("Presión", f"{presion} PSI")

# ---------- Predicción ----------
entrada = pd.DataFrame(
    [[temperatura, vibracion, horas, presion]],
    columns=["temperatura_c", "vibracion_mms", "horas_operacion", "presion_psi"],
)
estado = int(modelo.predict(entrada)[0])
probas = modelo.predict_proba(entrada)[0]
prob_estado = float(probas[list(modelo.classes_).index(estado)])

st.subheader("Diagnóstico del modelo ML")
mensaje = f"**{ESTADOS[estado]}** — Probabilidad: {prob_estado * 100:.1f}%"
if estado == 0:
    st.success(f"🟢 {mensaje}")
elif estado == 1:
    st.warning(f"🟡 {mensaje}")
else:
    st.error(f"🔴 {mensaje}")

with st.expander("Ver probabilidades por clase"):
    for clase, p in zip(modelo.classes_, probas):
        st.write(f"{ESTADOS[int(clase)]}: {p * 100:.1f}%")
        st.progress(float(p))

# ---------- Groq ----------
st.subheader("📝 Orden de Trabajo (IA Generativa)")


def generar_orden(api_key):
    client = Groq(api_key=api_key)
    modelos_activos = [
        m.id
        for m in client.models.list().data
        if not any(x in m.id.lower() for x in
                   ["whisper", "guard", "tts", "playai", "orpheus", "embed", "compound", "safeguard"])
    ]
    # Candidatos: primero los llama, luego el resto (se prueban en orden)
    candidatos = [m for m in modelos_activos if "llama" in m.lower()]
    candidatos += [m for m in modelos_activos if m not in candidatos]

    system_prompt = (
        "Eres un Jefe de Mantenimiento Industrial con 20 años de experiencia en plantas "
        "de manufactura automatizada. Respondes en español, de forma técnica, clara y concisa."
    )

    datos = (
        f"Lecturas: temperatura {temperatura} °C, vibración {vibracion} mm/s, "
        f"{horas} horas de operación, presión {presion} PSI. "
        f"Estado clasificado por el modelo: {estado} ({ESTADOS[estado]}) "
        f"con probabilidad {prob_estado * 100:.1f}%."
    )

    if estado == 0:
        instruccion = (
            f"{datos}\nEmite un dictamen BREVE (máximo 3 líneas) de monitoreo preventivo "
            "indicando que la operación es normal y qué seguir vigilando."
        )
    else:
        urgencia = "URGENTE, con protocolo de paro inmediato" if estado == 2 else "preventiva programada"
        instruccion = (
            f"{datos}\nGenera una Orden de Trabajo {urgencia} en formato Markdown con estas secciones:\n"
            "1. **Folio** (formato OT-AAAA-XXXX)\n"
            "2. **Diagnóstico**\n"
            "3. **Acciones Técnicas** (exactamente 3, numeradas)\n"
            "4. **Equipo de Seguridad (EPP)** requerido"
        )

    ultimo_error = None
    for modelo_groq in candidatos:
        try:
            resp = client.chat.completions.create(
                model=modelo_groq,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": instruccion},
                ],
                temperature=0.4,
            )
            return resp.choices[0].message.content, modelo_groq
        except Exception as e:  # modelo no válido para chat: probar el siguiente
            ultimo_error = e
    raise ultimo_error if ultimo_error else RuntimeError("Sin modelos de chat disponibles")


if st.button("Generar Orden de Trabajo", type="primary"):
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = None
        st.warning("⚠️ No se encontró GROQ_API_KEY en st.secrets. Se muestra solo el diagnóstico de ML.")

    if api_key:
        try:
            with st.spinner("Generando con Groq..."):
                texto, modelo_usado = generar_orden(api_key)
            st.markdown(texto)
            st.caption(f"Modelo Groq utilizado: {modelo_usado}")
        except Exception as e:
            st.error(f"No se pudo contactar a Groq ({type(e).__name__}). "
                     "Se mantiene el diagnóstico de ML.")
            st.caption(f"Detalle técnico: {str(e)[:300]}")
            st.info(f"Diagnóstico ML: {ESTADOS[estado]} (probabilidad {prob_estado * 100:.1f}%).")