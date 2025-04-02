# Image Caption Generator

## Overview
This project is a Streamlit-based web application that generates captions for uploaded images using a trained deep learning model. The application loads a pre-trained image captioning model and processes images to generate descriptive captions.

## Features
- **Multiple Image Uploads**: Users can upload multiple images at once.
- **Deep Learning-based Captioning**: Uses a CNN-LSTM-based model to generate captions.
- **Automated Model Download**: Downloads model files from GitHub releases if they are not available locally.
- **User-friendly Interface**: Built with Streamlit for easy interaction.

## Technologies Used
- **Python**
- **Streamlit** (for web interface)
- **TensorFlow/Keras** (for deep learning model loading and prediction)
- **NumPy & Matplotlib** (for image processing and visualization)
- **Pickle** (for loading the tokenizer)

## Installation & Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd image-caption-generator
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Model Files
The application downloads the following model files from GitHub if they are not available locally:
- `model.keras` (Trained caption generation model)
- `tokenizer.pkl` (Tokenizer for text processing)
- `feature_extractor.keras` (CNN-based image feature extractor)

## Usage
1. Open the application in a browser.
2. Upload one or multiple images.
3. The model generates captions for each image.
4. The uploaded images and their generated captions are displayed.

## License
This project is open-source and available under the MIT License.

## Author
[Your Name]

