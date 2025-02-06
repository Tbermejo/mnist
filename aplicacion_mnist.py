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
     st.set_page_config(page_title="Clasificación MNIST", layout="wide")
    st.title("🖼️ Clasificación de imágenes MNIST")
    st.write("""
    ### Selección del Mejor Modelo Mediante Búsqueda de Hiperparámetros en diferentes métodos de clasificación.

    En este análisis, se implementó un proceso iterativo para encontrar el mejor modelo de clasificación utilizando el Clasificador Naive Bayes (GaussianNB)**, Árboles de decisión (DecisionTreeClassifier)**, Máquinas de Soporte Vectorial (SVC)**, entre otros. Para ello, se realizó una búsqueda de hiperparámetros mediante la técnica de **Grid Search con validación cruzada**, con el objetivo de optimizar la precisión del modelo y mejorar su capacidad de generalización.

    Una vez identificado el mejor conjunto de hiperparámetros para el modelo: \n
    **Datos de test:**  0.9175 \n
    **Datos de train:**  0.9152 \n
    **Mejor hiperparámetro:** {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'} \n
    **Mejor precisión:** 0.907 \n
    **Kernel utilizado:** rbf \n
    El modelo óptimo fue entrenado con la totalidad de los datos de entrenamiento y evaluado sobre el conjunto de prueba. Para medir su desempeño, se calcularon métricas clave como la **precisión (accuracy)**, además de visualizar su comportamiento mediante una **matriz de confusión y la curva ROC**.""")

    # Cargar imágenes
    img1 = Image.open("ACC_SVC.png").resize((300, 300))
    img2 = Image.open("ROC curve SVC.png").resize((300, 300))

    # Mostrar imágenes en dos columnas
    col1, col2 = st.columns(2)
    with col1:
        st.image(img1, caption="Matriz de confusión", use_container_width=True)
    with col2:
        st.image(img2, caption="Curva ROC", use_container_width=True)
    
    st.markdown("### Sube una imagen y el modelo la clasificará en una de las 10 categorías del dataset MNIST.")
    st.write("""⬅️Ahora intenta clasificar tus imágenes en la barra lateral izquierda.""")
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
            st.sidebar.success(f"🔢 La imagen fue clasificada como: '{prediction}'.")

if __name__ == "__main__":
    main()


