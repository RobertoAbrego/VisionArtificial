import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
datos, metadatos = tfds.load('cats_vs_dogs', as_supervised = True, with_info = True);
print(datos, metadatos)
tfds.as_dataframe(datos['train'].take(5), metadatos)
for i, (imagen , etiqueta) in enumerate(datos['train'].take(2)):
    plt.subplot(1,2, 1 + i)
    plt.imshow(imagen)
