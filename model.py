import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# load data
data = keras.datasets.fashion_mnist

# split data
(train_images, train_labels), (test_images, test_labels) = data.load_data()

# define list of class names and pre-process images (by dividing each image by 255. Since each image is
# greyscale we are simply scaling the pixel values down to make computations easier for our model.)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# if you want to show image
# plt.imshow(train_images[1])
# plt.show()

train_images = train_images/255.0
test_images = test_images/255.0

# create neural network
model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28,28)),
	keras.layers.Dense(128, activation="relu"),
	keras.layers.Dense(10, activation="softmax")
	])

# training the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_images, train_labels, epochs=5)

# testing the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
# accuracy
print('\nTest accuracy:', test_acc)

# using the model
# predictions
predictions = model.predict(test_images)
# display first 5 images and their predictions
plt.figure(figsize=(5, 5))
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual : " + class_names[test_labels[i]])
    plt.title("Prediction : " + class_names[np.argmax(predictions[i])])
    plt.show()