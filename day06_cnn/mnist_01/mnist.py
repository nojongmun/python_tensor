import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

for i in train_images[100]:
    for col in i:
        print('%10f' % col, end='')
    print('\n')
print('\n')

plt.figure(figsize=(5,5))
image = train_images[100]
plt.imshow(image)
plt.show()
