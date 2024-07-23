Application de Reconnaissance de Chat
Cette application utilise un modèle de réseau de neurones pour reconnaître les chats dans les images téléchargées par l'utilisateur.

Prérequis
Python 3.x
TensorFlow
Flask
Un modèle pré-entraîné (chat_recognition_model.h5)

Installation
Clonez le dépôt: git clone https://github.com/votre-utilisateur/chat-recognition-app.git
cd chat-recognition-app
Créez un environnement virtuel et activez-le:

python -m venv venv
source venv/bin/activate  # Pour Linux/Mac
venv\Scripts\activate  # Pour Windows
Installez les dépendances:


pip install -r requirements.txt
Assurez-vous que le fichier chat_recognition_model.h5 est placé dans le répertoire model.

Utilisation
Lancez l'application Flask:

flask run
Ouvrez votre navigateur et allez à http://127.0.0.1:5000.

Téléchargez une image de chat et laissez l'application faire la prédiction.

Structure des fichiers
app.py: Le fichier principal contenant le code de l'application Flask.
templates/index.html: Le modèle HTML pour l'interface utilisateur.
model/chat_recognition_model.h5: Le modèle de réseau de neurones pré-entraîné.
uploads/: Le répertoire où les images téléchargées sont stockées.
requirements.txt: La liste des dépendances Python.
