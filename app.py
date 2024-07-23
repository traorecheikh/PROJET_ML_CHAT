from flask import Flask, request, render_template, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Charger le modèle pré-entraîné
model = load_model('model/chat_recognition_model.h5')

# S'assurer que le dossier des uploads existe
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    # Afficher la page d'accueil avec un résultat vide
    return render_template('index.html', result={})

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        # Pas de fichier image envoyé
        return render_template('index.html', result={"label": "Aucune image téléchargée.", "confidence": "", "image_url": ""})

    img_file = request.files['image']
    if img_file.filename == '':
        # Aucun fichier sélectionné
        return render_template('index.html', result={"label": "Aucun fichier sélectionné.", "confidence": "", "image_url": ""})

    img_path = os.path.join(UPLOAD_FOLDER, img_file.filename)
    img_file.save(img_path)

    # Charger et prétraiter l'image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Faire la prédiction
    prediction = model.predict(img_array)
    result_label = 'Chat' if prediction[0] > 0.5 else 'Pas Un Chat'
    result_confidence = round(prediction[0][0] * 100, 2) if result_label == 'Chat' else 100 - round(prediction[0][0] * 100, 2)

    return render_template('index.html', result={
        'label': result_label,
        'confidence': result_confidence,
        'image_url': img_file.filename
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # Retourner le fichier uploadé
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
