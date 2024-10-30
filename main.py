# %% [importar bibliotecas]
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# %% [Pré-processamento da Imagem]
# Caminho para a imagem que você quer usar
img_path = "dog.jpg"


# Função para carregar e pré-processar a imagem
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img


# Carrega e pré-processa a imagem
img = preprocess_image(img_path)

# %% [Configurar o Modelo e as Camadas para o Deep Dream]
# Carrega o modelo InceptionV3
base_model = inception_v3.InceptionV3(weights="imagenet", include_top=False)

# Escolhe camadas intermediárias para amplificar
dream_layers = [
    base_model.get_layer(name).output
    for name in ["mixed2", "mixed3", "mixed4", "mixed5"]  # Adicionando mixed5
]
dream_model = Model(inputs=base_model.input, outputs=dream_layers)


# %% [Configurar o Loss e as Funções de Gradiente]
# Calcula a perda
def calculate_loss(img, model):
    layer_activations = model(img)
    losses = [tf.reduce_mean(act) for act in layer_activations]
    return tf.reduce_sum(losses)


# Função de gradiente
@tf.function
def deepdream_step(img, model, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = calculate_loss(img, model)
    grads = tape.gradient(loss, img)
    grads = tf.math.l2_normalize(grads)
    img = img + grads * learning_rate
    img = tf.clip_by_value(img, -1, 1)
    return img, loss


# %% [Função para o Deep Dream e Visualização]
# Função Deep Dream
def run_deep_dream(
    img, model, steps=200, learning_rate=0.02
):  # Aumentando steps e learning_rate
    img = tf.convert_to_tensor(img)
    for step in range(steps):
        img, loss = deepdream_step(img, model, learning_rate)
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss.numpy()}")
    return img


# Aplica o efeito Deep Dream
dream_img = run_deep_dream(img, dream_model)


# Função para processar a imagem final
def deprocess_image(img):
    img = 255 * (img + 1.0) / 2.0
    return tf.cast(img, tf.uint8)


# Exibe a imagem final
plt.figure(figsize=(10, 10))
plt.imshow(deprocess_image(dream_img[0]).numpy())
plt.axis("off")
plt.show()
