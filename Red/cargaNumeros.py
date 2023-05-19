import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from tensorflow import keras

# Cargar el conjunto de datos MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocesar los datos
x_train = np.expand_dims(x_train, axis=-1) / 255.0
x_test = np.expand_dims(x_test, axis=-1) / 255.0
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Crear un generador de imágenes con rotación aleatoria
datagen = ImageDataGenerator(rotation_range=20,
                             width_shift_range=0.3,
                             height_shift_range=0.3,
                             shear_range=15,
                             zoom_range=[0.5, 1.5],
                             vertical_flip=True,
                             horizontal_flip=True)

# Generar lotes de imágenes rotadas y sus etiquetas
rotated_data = datagen.flow(x_train, y_train, batch_size=9, shuffle=False)

# Obtener las imágenes y las etiquetas del primer lote
rotated_images, rotated_labels = rotated_data.next()

# Visualizar las imágenes rotadas con sus etiquetas
fig, axs = plt.subplots(3, 3, figsize=(10, 10))
axs = axs.ravel()

for i in range(9):
    image = rotated_images[i]
    label = rotated_labels[i]

    axs[i].imshow(image.squeeze(), cmap='gray')
    axs[i].set_title('Label: {}'.format(np.argmax(label)))
    axs[i].axis('off')

plt.tight_layout()
plt.show()

# Definir el modelo
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compilar el modelo
print("Compilar modelo")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
print("Entrenamiento...")
model.fit(x_train, y_train, epochs=7, batch_size=32, validation_data=(x_test, y_test))

# Guardar los pesos del modelo
model.save('C:/Alejandro/Programacion/Python/calculadoraIA/Modelos/numeros.h5')
print("Modelo Guardado")
