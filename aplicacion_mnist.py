
import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import gzip
import pickle

def preprocess_image(image):
    image = image.convert('L')  # Convertir a escala de grises
    image = image.resize((28, 28))
    image_array = img_to_array(image) / 255.0
    image_array = image_array.reshape(1, 28 * 28)  # Ajustar dimensiones para el modelo
    return image_array

def load_model():
    filename = "model_trained_classifier.pkl.gz"
    with gzip.open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def main():
    st.set_page_config(page_title="Clasificación MNIST", layout="wide")
    st.title("🖼️ Clasificación de imágenes MNIST")
    st.markdown("### Sube una imagen y el modelo la clasificará en una de las 10 categorías del dataset MNIST.")
    
    st.sidebar.header("Carga de Imagen")
    uploaded_file = st.sidebar.file_uploader("Selecciona una imagen (PNG, JPG, JPEG):", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Imagen original", use_container_width=True)
        
        preprocessed_image = preprocess_image(image)
        
        with col2:
            st.image(image.convert('L').resize((28, 28)), caption="Imagen preprocesada", use_container_width=True)
        
        if st.sidebar.button("Clasificar imagen"):
            model = load_model()
            prediction = model.predict(preprocessed_image)
            predicted_label = np.argmax(prediction)  # Obtener la clase con mayor probabilidad
            st.sidebar.success(f"🔢 La imagen fue clasificada como: {predicted_label}, que corresponde al número '{predicted_label}'.")

if __name__ == "__main__":
    main()
