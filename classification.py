import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt
import sys
import numpy as np

# Carregar a imagem DICOM
image = tfio.image.decode_dicom_image(tf.io.read_file(sys.argv[1]))

# Exibir a imagem
plt.imshow(tf.squeeze(image), cmap='gray')
plt.title('Mamografia DICOM')
plt.savefig('mamografia.png')

# Pré-processar a imagem
image = tf.image.resize(image, [224, 224])  # Ajustar para o tamanho do modelo
image = tf.image.grayscale_to_rgb(image)  # Converter para RGB

# Remover dimensões extras, se necessário
image = tf.squeeze(image)  # Remove dimensões de tamanho 1
image = tf.expand_dims(image, axis=0)  # Adicionar dimensão do batch
image = image / 255.0  # Normalizar a imagem

# Verificar a forma da imagem
print("Forma da imagem:", image.shape)  # Deve ser (1, 224, 224, 3)

# Carregar um modelo pré-treinado (por exemplo, MobileNetV2)
base_model = tf.keras.applications.MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Congelar camadas do modelo pré-treinado

# Adicionar camadas de classificação
inputs = tf.keras.layers.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)  # Adicionar `training=False` para evitar dropout
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)  # Para classificação binária
classification_model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compilar o modelo
classification_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Classificar a imagem
predictions = classification_model.predict(image)
predicted_class = np.round(predictions[0][0])  # Para classificação binária (0 ou 1)

if predicted_class == 1:
    print("A mamografia é suspeita.")
else:
    print("A mamografia é normal.")
