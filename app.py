import streamlit as st
from ultralytics import YOLO
from PIL import Image
import time

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Detector de Somatotipos",
    page_icon="💪",
    layout="centered"
)

# --- ESTILOS CSS (Para que se vea más bonito) ---
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- TÍTULO Y DESCRIPCIÓN ---
st.title("🧬 AI Somatotype Scanner")
st.write("Sube una foto de cuerpo completo (preferiblemente en ropa ajustada o ropa interior) para analizar tu genética.")

# --- CARGAR EL MODELO ---
# Usamos @st.cache_resource para que no recargue el modelo cada vez que tocas un botón
@st.cache_resource
def load_model():
    try:
        # Asegúrate de que 'best.pt' esté en la misma carpeta
        model = YOLO('best.pt') 
        return model
    except Exception as e:
        st.error(f"No se encontró el archivo del modelo: {e}")
        return None

model = load_model()

# --- SUBIDA DE IMAGEN ---
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None and model is not None:
    # 1. Mostrar la imagen subida
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen subida', use_container_width=True)
    
    # Botón de análisis
    if st.button('🔍 Analizar Somatotipo'):
        with st.spinner('Analizando geometría corporal...'):
            # Simular un pequeño tiempo de carga para efecto visual
            time.sleep(1) 
            
            # 2. Realizar la predicción
            results = model(image)
            
            # 3. Obtener resultados
            # YOLO devuelve una lista, tomamos el primer elemento
            res = results[0]
            probs = res.probs  # Probabilidades
            
            # Obtener la clase ganadora y su confianza
            top1_index = probs.top1
            top1_label = res.names[top1_index].upper() # Nombre (Ecto, Meso, Endo)
            top1_conf = probs.top1conf.item() # Confianza (0 a 1)

        # --- MOSTRAR RESULTADOS ---
        st.success("¡Análisis completado!")
        
        st.markdown(f"""
        <div style="text-align: center;">
            <h2>Tu somatotipo dominante es:</h2>
            <h1 style="color: #FF4B4B; font-size: 50px;">{top1_label}</h1>
            <p>Confianza del modelo: <b>{top1_conf * 100:.1f}%</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Barras de probabilidad para todas las clases
        st.write("---")
        st.subheader("Desglose detallado:")
        
        # Iterar sobre todas las clases para mostrar barras
        for i, class_name in res.names.items():
            probabilidad = probs.data[i].item()
            st.write(f"**{class_name.capitalize()}**")
            st.progress(probabilidad)
            st.caption(f"{probabilidad * 100:.1f}%")

        # --- RECOMENDACIONES PERSONALIZADAS (Opcional) ---
        st.write("---")
        st.subheader("💡 Recomendación rápida")
        
        if "ECTO" in top1_label:
            st.info("**Consejo Ectomorfo:** Tu metabolismo es rápido. Prioriza el consumo de carbohidratos complejos y entrenamiento de fuerza con descansos largos.")
        elif "MESO" in top1_label:
            st.info("**Consejo Mesomorfo:** Tienes facilidad para ganar músculo. Un balance entre pesas y cardio moderado te dará resultados increíbles.")
        elif "ENDO" in top1_label:
            st.info("**Consejo Endomorfo:** Tiendes a acumular energía. Prioriza las proteínas, controla los carbohidratos y mantén una actividad física constante (HIIT o cardio).")

else:
    if model is None:
        st.warning("⚠️ Por favor, coloca el archivo 'best.pt' en la misma carpeta que este script.")