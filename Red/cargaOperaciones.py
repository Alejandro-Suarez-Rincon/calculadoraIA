import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

# Cargar el conjunto de datos MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar los valores de píxel en el rango [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Expandir las dimensiones de los datos de entrada
x_train = tf.expand_dims(x_train, axis=-1)
x_test = tf.expand_dims(x_test, axis=-1)

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

# Definir el modelo de la red neuronal convolucional
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Guardar el modelo en el archivo 'operaciones.h5'
model.save('C:/Modelos/operaciones.h5')

print("Modelo guardado en 'operaciones.h5'")