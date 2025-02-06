import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import gzip
import pickle

# Preprocesar la imagen
def preprocess_image(image):
    image = image.convert('L')  # Convertir a escala de grises
    image = image.resize((28, 28))  # Redimensionar la imagen a 28x28
    image_array = img_to_array(image) / 255.0  # Normalizar los valores de píxeles
    image_array = np.expand_dims(image_array, axis=0)  # Expandir dimensiones para la predicción
    return image_array

# Cargar el modelo entrenado
def load_model():
    filename = "model_trained_classifierSVC.pkl.gz"
    with gzip.open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

# Función principal
def main():
    st.title("Clasificación de la base de datos MNIST")
    st.markdown("Sube una imagen para clasificar")

    uploaded_file = st.file_uploader("Selecciona una imagen (PNG, JPG, JPEG):", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen subida", use_column_width=True)  # Mostrar imagen subida

        # Preprocesar la imagen
        preprocessed_image = preprocess_image(image)

        # Mostrar imagen preprocesada en la barra lateral
        st.sidebar.image(preprocessed_image[0], caption="Imagen preprocesada", use_column_width=True)

        if st.button("Clasificar imagen"):
            st.markdown("Imagen clasificada")
            model = load_model()

            # Realizar la predicción
            prediction = model.predict(preprocessed_image.reshape(1, -1))  # Redimensionar para el modelo (1, 784)
            st.markdown(f"La imagen fue clasificada como: {prediction[0]}")  # Mostrar la predicción

if __name__ == "__main__":
    main()


