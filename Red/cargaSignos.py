import os

import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras

# Ruta del directorio que contiene las imágenes del dataset
dataset_dir = "C:\Alejandro\Programacion\Python\calculadoraIA\Modelos\DataSetOperaciones"

# Número de clases en el dataset
num_classes = 2

# Obtener la cantidad de imágenes en el dataset
image_count = sum([len(files) for _, _, files in os.walk(dataset_dir)])

# Obtener el tamaño de las imágenes del dataset
image_size = (28, 28)

# Preprocesar los datos del dataset
datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=image_size,
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=32,
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=image_size,
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=32,
    subset='validation'
)

# Obtener la cantidad de pasos por época para el entrenamiento y la validación
steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = validation_generator.samples // validation_generator.batch_size

# Obtener las imágenes de entrenamiento
train_images, train_labels = next(train_generator)

# Visualizar las imágenes
fig, axes = plt.subplots(4, 8, figsize=(12, 8))
axes = axes.flatten()

for i, ax in enumerate(axes):
    ax.imshow(train_images[i].reshape(image_size), cmap='gray')
    ax.axis('off')
    ax.set_title(f"Label: {train_labels[i]}")

plt.tight_layout()
plt.show()

# Definir el modelo
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(2, activation='softmax')# layers de salida
])

# Compilar el modelo
print("Compilar modelo")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)

# Entrenar el modelo con el dataset local
print("Entrenamiento...")
model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=7,
    validation_data=validation_generator,
    validation_steps=validation_steps
)

# Guardar los pesos del modelo entrenado
model.save('C:\Alejandro\Programacion\Python\calculadoraIA\Modelos\signos.h5')
print("Modelo Guardado")
