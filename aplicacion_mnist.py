
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
    filename = "model_trained_classifierSVC.pkl.gz"
    with gzip.open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def main():
    st.set_page_config(page_title="Clasificación MNIST", layout="wide")
    st.title("🖼️ Clasificación de imágenes MNIST")
    st.write("""
    ### Selección del Mejor Modelo Mediante Búsqueda de Hiperparámetros en diferentes métodos de clasificación.

    En este análisis, se implementó un proceso iterativo para encontrar el mejor modelo de clasificación utilizando el Clasificador Naive Bayes (`GaussianNB`)**, Árboles de decisión (`DecisionTreeClassifier`)**, **Máquinas de Soporte Vectorial (SVM)**, entre otros. Para ello, se realizó una búsqueda de hiperparámetros mediante la técnica de **Grid Search con validación cruzada**, con el objetivo de optimizar la precisión del modelo y mejorar su capacidad de generalización.

    Una vez identificado el mejor conjunto de hiperparámetros, el modelo óptimo fue entrenado con la totalidad de los datos de entrenamiento y evaluado sobre el conjunto de prueba. Para medir su desempeño, se calcularon métricas clave como la **precisión (accuracy)**, además de visualizar su comportamiento mediante una **matriz de confusión y la curva ROC**.""")

    # Agregar imagen desde un archivo local
    st.image("ACC_SVC.png", caption="Matriz de confusión", use_column_width=True)

    import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import load_digits

# Cargar un dataset de ejemplo (puedes reemplazar esto con tus propios datos)
digits = load_digits()
X = digits.data
y = digits.target

# Convertir el problema de clasificación múltiple en uno binario
y_binary = (y == 1).astype(int)  # Ejemplo: clasificar "1" como positivo y el resto como negativo

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42)

# Entrenar el modelo (SVM en este caso)
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Obtener las probabilidades de las predicciones
y_prob = model.predict_proba(X_test)[:, 1]  # Probabilidad de la clase positiva

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Visualizar la curva ROC con Matplotlib
fig, ax = plt.subplots()
ax.plot(fpr, tpr, color='blue', lw=2, label='Curva ROC (área = %0.2f)' % roc_auc)
ax.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Línea diagonal
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('Tasa de falsos positivos (FPR)')
ax.set_ylabel('Tasa de verdaderos positivos (TPR)')
ax.set_title('Curva ROC')
ax.legend(loc='lower right')

# Mostrar la curva ROC en Streamlit
st.pyplot(fig)


    
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
