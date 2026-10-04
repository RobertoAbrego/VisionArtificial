import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
datos, metadatos = tfds.load('cats_vs_dogs', as_supervised = True, with_info = True);
#print(datos, metadatos)
#tfds.as_dataframe(datos['train'].take(5), metadatos)

for imagen, etiqueta in datos['train'].take(5):
    plt.figure()
    plt.imshow(imagen)
    plt.title("Perro" if etiqueta.numpy() == 1 else "Gato")
    plt.axis("off")
    plt.show()