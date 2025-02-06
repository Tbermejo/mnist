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
    filename = "model_trained_classifier_SVC_MinMaxScaler.pkl.gz"
    try:
        with gzip.open(filename, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

def main():
    st.set_page_config(page_title="Clasificación MNIST", layout="wide")
    st.title("🖼️ Clasificación de imágenes MNIST")
    st.write("""
    ### Selección del Mejor Modelo Mediante Búsqueda de Hiperparámetros en diferentes métodos de clasificación.
    ... (tú ya sabes qué poner aquí) ...
    """)

    # Cargar imágenes de la matriz de confusión y curva ROC
    try:
        img1 = Image.open("ACC_SVC.png").resize((300, 300))
        img2 = Image.open("ROC curve SVC.png").resize((300, 300))
    except FileNotFoundError:
        st.error("No se encontraron las imágenes necesarias para mostrar la matriz de confusión y la curva ROC.")
        return

    # Mostrar imágenes en dos columnas
    col1, col2 = st.columns(2)
    with col1:
        st.image(img1, caption="Matriz de confusión", use_container_width=True)
    with col2:
        st.image(img2, caption="Curva ROC", use_container_width=True)

    st.markdown("### Sube una imagen y el modelo la clasificará en una de las 10 categorías del dataset MNIST.")
    st.write("⬅️ Ahora intenta clasificar tus imágenes en la barra lateral izquierda.")
    st.sidebar.header("Carga de Imagen")
    uploaded_file = st.sidebar.file_uploader("Selecciona una imagen (PNG, JPG, JPEG):", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        # Mostrar la imagen original y preprocesada
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Imagen original", use_container_width=True)

        preprocessed_image = preprocess_image(image)

        with col2:
            st.image(image.convert('L').resize((28, 28)), caption="Imagen preprocesada", use_container_width=True)

        # Clasificar imagen
        if st.sidebar.button("Clasificar imagen"):
            model = load_model()
            prediction = model.predict(preprocess_image)
            st.sidebar.success(f"🔢 La imagen fue clasificada como: '{prediction}'.") 

if __name__ == "__main__":
    main()

