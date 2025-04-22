import streamlit as st
from tensorflow.keras.models import load_model  # Usamos load_model en lugar de model_from_json
from PIL import Image, UnidentifiedImageError
import numpy as np
import os

# Configuración de la página
st.set_page_config(page_title="Classificador Gats vs Gossos", layout="centered")
st.title("🐶 Classificador de Gossos i Gats 🐱")
st.markdown("Puja una imatge i la IA et dirà si veu un gos o un gat! �")

# Subida de archivos
uploaded_file = st.file_uploader("📤 Pujar imatge (jpg, png)", type=["jpg", "jpeg", "png"])

# Verificación del modelo
if not os.path.exists("model_gats_gossos.h5"):
    st.error("❌ Error: No s'ha trobat el fitxer 'model_gats_gossos.h5'. Assegura't que:")
    st.error("1. El fitxer està al mateix directori que app.py")
    st.error("2. El nom del fitxer és EXACTAMENT 'model_gats_gossos.h5'")
else:
    try:
        # Cargar el modelo
        model = load_model("model_gats_gossos.h5")
        
        if uploaded_file:
            try:
                # Procesar imagen
                image = Image.open(uploaded_file).convert("RGB").resize((100, 100))
                st.image(image, caption='📷 Imatge pujada', use_column_width=True)
                
                # Preprocesamiento
                img_array = np.array(image) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Predicción
                prediction = model.predict(img_array)
                prob = float(prediction[0])
                
                # Mostrar resultado
                if prob > 0.5:
                    st.success(f"Resultat: És un **gos** 🐶 amb {prob*100:.2f}% de confiança!")
                    st.balloons()
                else:
                    st.success(f"Resultat: És un **gat** 🐱 amb {(1-prob)*100:.2f}% de confiança!")
                    st.balloons()
                    
            except UnidentifiedImageError:
                st.error("❌ Error: No s'ha pogut llegir la imatge. Posa un format vàlid (jpg, png).")
            except Exception as e:
                st.error(f"❌ Error inesperat en processar la imatge: {str(e)}")
                
    except Exception as e:
        st.error(f"❌ Error en carregar el model: {str(e)}")
        st.error("Recomanacions:")
        st.error("1. Verifica que el model està en format .h5")
        st.error("2. Assegura't que tens la versió correcta de TensorFlow instal·lada")
