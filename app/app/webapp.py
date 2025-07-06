import streamlit as st
import pandas as pd
from sklearn.externals import joblib  # Para cargar tu modelo previamente entrenado
import os

# Cargar el modelo previamente entrenado
modelo = joblib.load('ruta_al_modelo/modelo_entrenado.pkl')  # Reemplaza con la ruta correcta

# Título de la aplicación
st.title('Predicción de modelo ML')

# Subida de archivo
st.subheader('Sube tu archivo CSV')

# Permitir al usuario subir un archivo CSV
archivo_subido = st.file_uploader('Elige un archivo CSV', type=['csv'])

if archivo_subido is not None:
    # Leer el archivo CSV
    df = pd.read_csv(archivo_subido)

    # Mostrar los primeros datos del archivo
    st.write(df.head())

    # Preprocesar los datos si es necesario, dependiendo de tu modelo
    # Por ejemplo, si tu modelo necesita que los datos sean normalizados o procesados de alguna manera:
    # df = preprocesar(df)  # Reemplaza con tu función de preprocesado
    
    # Realizar la predicción
    predicciones = modelo.predict(df)  # Esto depende de cómo hayas entrenado tu modelo

    # Mostrar las predicciones
    st.subheader('Predicciones')
    st.write(predicciones)

    # Si las predicciones son un conjunto de datos con muchas columnas
    # Puedes visualizarlas de una mejor forma, por ejemplo:
    st.dataframe(pd.DataFrame(predicciones, columns=['Predicción']))

else:
    st.warning('Por favor sube un archivo para realizar la predicción.')

