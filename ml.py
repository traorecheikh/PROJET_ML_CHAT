from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import os

print("All imports are successful!")

# verifier le repertoire
train_dir = 'data/train'
val_dir = 'data/validation'

if not os.path.exists(train_dir):
    print(f"Training directory {train_dir} does not exist.")
if not os.path.exists(val_dir):
    print(f"Validation directory {val_dir} does not exist.")

# charger modele initiale
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
# classification binaire
x = Dense(1, activation='sigmoid')(x)

# definiyion du model
model = Model(inputs=base_model.input, outputs=x)

# geler le modele
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# generer donnee entrainements
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=20, zoom_range=0.2)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary'
)
#771758304
# generer donnee validation
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary'
)

# Train the model with verbose output
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    verbose=2
)

# Enregistrer model (duree approximative 1h20mn)
model.save('model/chat_recognition_model.h5')

print("Apprentissage reussi et enregistrer dans le repertoire model!")
