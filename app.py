import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, UnidentifiedImageError
import numpy as np
import os

# Configuració de la pàgina
st.set_page_config(
    page_title="Classificador Gats vs Gossos",
    page_icon="🐾",
    layout="centered"
)

# Títol i descripció
st.title("🐶 Classificador de Gossos i Gats 🐱")
st.markdown("""
Puja una imatge i la IA et dirà si és un gos o un gat!
*Format recomanat: 100x100px, fons neutre*
""")

# Carregar el model (un sol cop al iniciar)
@st.cache_resource
def load_my_model():
    try:
        return load_model("model_gats_gossos.h5")
    except Exception as e:
        st.error(f"❌ Error carregant el model: {str(e)}")
        return None

model = load_my_model()

# Widget per pujar imatges
uploaded_file = st.file_uploader(
    "📤 Pujar imatge",
    type=["jpg", "jpeg", "png"],
    help="Imatges amb format JPG, JPEG o PNG"
)

if uploaded_file and model:
    try:
        # Processament de la imatge
        image = Image.open(uploaded_file).convert("RGB").resize((100, 100))
        
        # Mostrar la imatge
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Imatge original", width=150)
        
        # Preprocessat
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predicció
        with st.spinner("Analitzant..."):
            prediction = model.predict(img_array)
            prob = float(prediction[0])
        
        # Resultat
        with col2:
            if prob > 0.5:
                st.success(f"**Gos** 🐶\nConfiança: {prob*100:.1f}%")
                st.balloons()
            else:
                st.success(f"**Gat** 🐱\nConfiança: {(1-prob)*100:.1f}%")
                st.snow()
        
        # Gràfic addicional
        st.progress(int(max(prob, 1-prob)*100))
        
    except UnidentifiedImageError:
        st.error("Format d'imatge no reconegut. Puja un arxiu JPG, JPEG o PNG vàlid.")
    except Exception as e:
        st.error(f"Error inesperat: {str(e)}")

# Missatge si no hi ha model
if not model:
    st.warning("""
    ⚠️ Model no disponible. Assegura't que:
    - `model_gats_gossos.h5` existeix al directori
    - El model és compatible amb TensorFlow 2.15
    """)
