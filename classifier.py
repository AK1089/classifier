# necessary imports for categorisation mode. tf is for machine learning,
# np for maths, plt for graphs, and tiktoken for tokenisation.
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

import tiktoken

# training and test data from the Fashion MNIST dataset
(xtrain, ytrain), (xtest, ytest) = keras.datasets.fashion_mnist.load_data()
print(f"Loaded {xtrain.shape[0]} training samples and {xtest.shape[0]} test samples.")

# Model structure: 784 -> 300 -> 100 -> 10 layers (266,610 parameters)
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
print(model.summary())

# normalises the data
xvalid, xtrain = xtrain[:5000]/255.0, xtrain[5000:]/255.0
yvalid, ytrain = ytrain[:5000], ytrain[5000:]

# creates and trains the model
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="Adam",
              metrics=["accuracy"])

history = model.fit(xtrain, ytrain, epochs=50, 
                    validation_data=(xvalid, yvalid))
