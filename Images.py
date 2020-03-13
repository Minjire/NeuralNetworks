import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# get data from keras
data = keras.datasets.fashion_mnist

# split our data into training and testing data to test the accuracy of the model on data
(train_images, train_labels), (test_images, test_labels) = data.load_data()

# list to define labels(0-9)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# normalize/shrink data down within a range, avoid large numbers
train_images = train_images/255.0
test_images = test_images/255.0

# create model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # flatten data (input layer)
    keras.layers.Dense(128, activation="relu"),  # dense layer--fully connected layer (hidden layer)
    keras.layers.Dense(10, activation="softmax")  # softmax to add up to 1 (output layer)
    ])

# Compiling the model is just picking the optimizer, loss function and metrics to keep track of
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# train model, epochs defines iteration of seeing a single input
model.fit(train_images, train_labels, epochs=5)

"""# test for model accuracy after training
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('\nTest accuracy:', test_acc)"""

prediction = model.predict(test_images)

# display the first 5 images and their predictions
plt.figure(figsize=(5, 5))
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()


