# %% [markdown]
# Pequeño proyecto de redes neuronales convolucionales para aprendizaje de perros y gatos.

# %%
%pip install -r requirements.txt #Instalar dependencias

# %%
#Importando librerias
import tensorflow as tf
import tensorflow_datasets as tfds      #Bancos de imagenes
import matplotlib.pyplot as plt         #Mostrar los datos
import cv2
import numpy as np
from tensorflow.keras.callbacks import TensorBoard #Tabla de estadisticas de redes neuronales

# %%
datos, metadatos = tfds.load('cats_vs_dogs', as_supervised = True, with_info = True);
#print(datos, metadatos)                                        #Existe nadamas para mostrar datos
tfds.as_dataframe(datos['train'].take(5), metadatos)           #Existe nadamas para mostrar datos

# %%
plt.figure(figsize=(20,20)) #Dimensiones
Tamano_img = 100
for i, (imagen , etiqueta) in enumerate(datos['train'].take(25)): #traer los datos
    plt.subplot(5,5,1+i) #Matriz 5x5
    plt.xticks([]) #limpiando las imagenes en x, (cosas de matplotlib)
    plt.yticks([]) #limpiando las imagenes en y, (cosas de matplotlib)
    imagen = imagen.numpy()
    imagen = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY) #Cambiar de color gris
    imagen = cv2.resize(imagen,(Tamano_img, Tamano_img))        #Redimensionar las imagenes a 200*200 px
    plt.imshow(imagen, cmap='gray') #Mostrar los datos y cmap para un solo canal

# %%
datos_entrenamiento = []
for i, (imagen, etiqueta) in enumerate(datos['train']): #todos los datos
    imagen = imagen.numpy()
    imagen = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)           #Cambiar de color gris
    imagen = cv2.resize(imagen,(Tamano_img, Tamano_img))        #Redimensionar las imagenes a 100*100 px
    imagen.reshape(100,100,1)                                   #Es una imagen de 100*100 * 1 canal de color
    datos_entrenamiento.append([imagen, etiqueta])

# %%
X = [] #Imagenes de entrada (Pixeles)
Y = [] #Etiquetas (Si es perro o gato)
import numpy as np
for imagen, etiqueta in datos_entrenamiento:
    X.append(imagen)
    Y.append(etiqueta)
X = np.array(X).astype(float)/255
Y = np.array(Y)
#X.shape         #23262 entradas (imagenes) -> 100 * 100 pixeles
#Y.shape         #23262 entradas (1 para perro y 0 para gato)

# %% [markdown]
# Modelos sin aumento de datos (No recomendado)

# %%
modeloDenso = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(100,100,1)), #Recibiria datos de 100 * 100, osea 1000 entradas y un solo canal de colores.
    tf.keras.layers.Dense(150, activation="relu"), # Consta de 150 neuronas y relu es que deja pasar los valores positivos sin cambios y convierte los valores negativos en cero.
    tf.keras.layers.Dense(150, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

modeloCNN = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(100,100,1)), 
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation="relu"), 
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation="relu"), 
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

modeloCNN2 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(100,100,1)), 
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation="relu"), 
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation="relu"), 
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(250, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# %% [markdown]
# Compilación de los modelos

# %%
modeloDenso.compile(optimizer = "adam",
                    loss= "binary_crossentropy",
                    metrics = ["accuracy"]
                    )
    

modeloCNN.compile(optimizer = "adam",
                    loss= "binary_crossentropy",
                    metrics = ["accuracy"]
                    )


modeloCNN2.compile(optimizer = "adam",
                    loss= "binary_crossentropy",
                    metrics = ["accuracy"]
                    )

# %% [markdown]
# Entrenamiento neuronal

# %%
TensorBoardDenso = TensorBoard(log_dir="logs/denso")
modeloDenso.fit(X, Y, batch_size=32, validation_split=0.15, epochs=100, callbacks=[TensorBoardDenso])


