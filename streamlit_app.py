import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Carregar o modelo de classificação de células (substitua pelo caminho do seu modelo)
# O modelo deve ser treinado para classificar basófilos, neutrófilos, leucócitos, etc.
try:
    model = load_model("modelo_leucograma.h5")  # Substitua pelo caminho do seu modelo
except Exception as e:
    st.error("Erro ao carregar o modelo. Verifique o caminho e o modelo treinado.")
    model = None

# Função para carregar e pré-processar a imagem
def load_and_preprocess_image(uploaded_image):
    image = Image.open(uploaded_image)
    image = image.convert("RGB")
    image = np.array(image)
    return image

# Função para processar a imagem e segmentar células
def process_image(image):
    # Redimensionar para facilitar a visualização e processamento
    image_resized = cv2.resize(image, (640, 800))
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)

    # Detectar contornos, que são candidatos a células
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cell_images = []
    for cnt in contours:
        # Filtrar pequenas áreas para evitar ruído
        area = cv2.contourArea(cnt)
        if area > 50:  # Ajuste o threshold de área conforme necessário
            x, y, w, h = cv2.boundingRect(cnt)
            cell_image = image_resized[y:y+h, x:x+w]
            cell_images.append(cell_image)
    
    return cell_images, image_resized, contours

# Função para classificar as células detectadas
def classify_cells(cell_images):
    cell_counts = {"Basófilo": 0, "Neutrófilo": 0, "Leucócito": 0, "Outros": 0}
    for cell_image in cell_images:
        # Redimensionar para o tamanho esperado pelo modelo
        cell_image_resized = cv2.resize(cell_image, (64, 64))  # Ajuste conforme o modelo
        cell_image_resized = np.expand_dims(cell_image_resized, axis=0) / 255.0  # Normalizar

        # Fazer a previsão
        if model is not None:
            prediction = model.predict(cell_image_resized)
            cell_type = np.argmax(prediction)  # Obter o índice da classe com maior probabilidade
            if cell_type == 0:
                cell_counts["Basófilo"] += 1
            elif cell_type == 1:    
                cell_counts["Neutrófilo"] += 1
            elif cell_type == 2:
                cell_counts["Leucócito"] += 1
            else:
                cell_counts["Outros"] += 1
        else:
            st.warning("Modelo não carregado corretamente. Apenas exibindo contornos.")
    
    return cell_counts

# Configurar a aplicação no Streamlit
st.title("Análise de Leucograma: Contagem de Basófilos, Neutrófilos e Leucócitos")
st.write("Carregue uma imagem de leucograma para análise das células.")

# Upload da imagem
uploaded_image = st.file_uploader("Faça upload de uma imagem (png, jpeg, jpg)", type=["png", "jpeg", "jpg"])

if uploaded_image is not None:
    # Carregar e pré-processar a imagem
    image = load_and_preprocess_image(uploaded_image)

    # Processar a imagem para detectar contornos de células
    cell_images, processed_image, contours = process_image(image)

    # Classificar as células detectadas
    cell_counts = classify_cells(cell_images)

    # Exibir a imagem original e processada
    st.image(image, caption="Imagem Original", use_column_width=True)
    st.write("### Resultados da Análise")
    for cell_type, count in cell_counts.items():
        st.write(f"{cell_type}: {count}")

    # Exibir a imagem processada com contornos das células
    for cnt in contours:
        cv2.drawContours(processed_image, [cnt], -1, (0, 255, 0), 1)
    st.image(processed_image, caption="Imagem Processada com Contornos de Células", use_column_width=True)
