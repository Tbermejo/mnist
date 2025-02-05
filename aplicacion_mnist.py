
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
    with gzip.open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def main():
    st.set_page_config(page_title="Clasificación MNIST", layout="wide")
    st.title("🖼️ Clasificación de imágenes MNIST")
    st.write("""
    ### Selección del Mejor Modelo Mediante Búsqueda de Hiperparámetros en diferentes métodos de clasificación.

    En este análisis, se implementó un proceso iterativo para encontrar el mejor modelo de clasificación utilizando el Clasificador Naive Bayes (`GaussianNB`)**, Árboles de decisión (`DecisionTreeClassifier`)**, **Máquinas de Soporte Vectorial (SVM)**, entre otros. Para ello, se realizó una búsqueda de hiperparámetros mediante la técnica de **Grid Search con validación cruzada**, con el objetivo de optimizar la precisión del modelo y mejorar su capacidad de generalización.

    Una vez identificado el mejor conjunto de hiperparámetros: \n
    **Datos de test:**  0.9175 \n
    **Datos de train:**  0.9152 \n
    **Mejor hiperparámetro:** {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'} \n
    **Mejor precisión:** 0.907 \n
    **Kernel utilizado:** rbf \n
    el modelo óptimo fue entrenado con la totalidad de los datos de entrenamiento y evaluado sobre el conjunto de prueba. Para medir su desempeño, se calcularon métricas clave como la **precisión (accuracy)**, además de visualizar su comportamiento mediante una **matriz de confusión y la curva ROC**.""")

    # Agregar imagen desde un archivo local
    col1, col2 = st.columns(2)

    col1, col2 = st.columns(2)
    with col1:
        st.image("ACC_SVC.png", caption="Matriz de confusión", use_container_width=True)
    with col2:
        st.image("ROC curve SVC.png", caption="Curva ROC", use_container_width=True)

    
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
            st.sidebar.success(f"🔢 La imagen fue clasificada como: '{prediction}'.")
            
            

if __name__ == "__main__":
    main()
