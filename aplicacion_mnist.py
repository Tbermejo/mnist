import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gzip
import pickle
from sklearn.metrics import confusion_matrix, roc_curve, auc
from io import BytesIO
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler

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

    En este análisis, se implementó un proceso iterativo para encontrar el mejor modelo de clasificación utilizando el Clasificador Naive Bayes (`GaussianNB`)**, Árboles de decisión (`DecisionTreeClassifier`)**, Máquinas de Soporte Vectorial (`SVC`)**, entre otros. Para ello, se realizó una búsqueda de hiperparámetros mediante la técnica de **Grid Search con validación cruzada**, con el objetivo de optimizar la precisión del modelo y mejorar su capacidad de generalización.

    Una vez identificado el mejor conjunto de hiperparámetros para el modelo: \n
    **Datos de test:**  0.9175 \n
    **Datos de train:**  0.9152 \n
    **Mejor hiperparámetro:** {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'} \n
    **Mejor precisión:** 0.907 \n
    **Kernel utilizado:** rbf \n
    El modelo óptimo fue entrenado con la totalidad de los datos de entrenamiento y evaluado sobre el conjunto de prueba. Para medir su desempeño, se calcularon métricas clave como la **precisión (accuracy)**, además de visualizar su comportamiento mediante una **matriz de confusión y la curva ROC**.""")

    #Cargar los datos de MNIST
    (X_train,y_train),(X_tests,y_test) = mnist.load_data()
    
    #Normalizar los datos
    X_train=X_train/255.0
    X_test=X_tests/255.0
    
    #Aplanar los datos
    X_train=X_train.reshape(60000,28*28)
    X_test=X_test.reshape(10000,28*28)

    #Aplicar MinMaxScaler a los datos para mejorar la precisión del modelo SVC
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #Cargar el modelo previamente entrenado
    filename = "model_trained_classifier_SVC_MinMaxScaler.pkl.gz"
    with gzip.open(filename, 'rb') as f:
        model = pickle.load(f)

    #Hacer predicciones sobre el conjunto de prueba
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]  # Probabilidades para la clase positiva

    #Calcular la matriz de confusión
    cm = confusion_matrix(y_test, y_pred)

    #Calcular la curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    #Graficar la matriz de confusión
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(10), yticklabels=np.arange(10))
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión')
    plt.show()

    #Graficar la curva ROC
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Línea de aleatoriedad
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curva ROC')
    plt.legend(loc='lower right')
    plt.show()

    #Convertir gráficos en imágenes para Streamlit
    def fig_to_image(fig):
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        return buf

    #Mostrar los gráficos en la aplicación (en dos columnas)
    col1, col2 = st.columns(2)
    with col1:
        fig_cm = plot_confusion_matrix(y_true, y_pred)
        st.image(fig_to_image(fig_cm), caption="Matriz de Confusión", use_container_width=True)

    with col2:
        fig_roc = plot_roc_curve(y_true, y_scores)
        st.image(fig_to_image(fig_roc), caption="Curva ROC", use_container_width=True)

    # Sección de carga de imágenes y clasificación
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
