import streamlit as st
import numpy as np
import requests
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import pickle

def download_file(url, filename):
    if not os.path.exists(filename):
        with requests.get(url, stream=True) as response:
            with open(filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

def generate_and_display_caption(image_path, model_path, tokenizer_path, feature_extractor_path, max_length=36, img_size=224):
    caption_model = load_model(model_path)
    feature_extractor = load_model(feature_extractor_path)

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    img = load_img(image_path, target_size=(img_size, img_size))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    image_features = feature_extractor.predict(img, verbose=0)

    in_text = "startseq"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = caption_model.predict([image_features, sequence], verbose=0)
        yhat_index = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat_index, None)
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break
    caption = in_text.replace("startseq", "").replace("endseq", "").strip()

    img = load_img(image_path, target_size=(img_size, img_size))
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title(caption, fontsize=16, color='blue')
    st.pyplot(plt)

def main():
    st.title("Image Caption Generator")
    st.write("Upload images and generate captions using the trained model.")

    model_url = "https://github.com/agnihotri-anxh/Image-Caption-Generator/releases/download/models/model.keras"
    tokenizer_url = "https://github.com/agnihotri-anxh/Image-Caption-Generator/releases/download/models/tokenizer.pkl"
    feature_extractor_url = "https://github.com/agnihotri-anxh/Image-Caption-Generator/releases/download/models/feature_extractor.keras"

    model_path = "model.keras"
    tokenizer_path = "tokenizer.pkl"
    feature_extractor_path = "feature_extractor.keras"

    download_file(model_url, model_path)
    download_file(tokenizer_url, tokenizer_path)
    download_file(feature_extractor_url, feature_extractor_path)

    uploaded_images = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_images:
        for uploaded_image in uploaded_images:
            image_path = f"uploaded_{uploaded_image.name}"
            with open(image_path, "wb") as f:
                f.write(uploaded_image.getbuffer())

            generate_and_display_caption(image_path, model_path, tokenizer_path, feature_extractor_path)

if __name__ == "__main__":
    main()
