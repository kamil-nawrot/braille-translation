import tensorflow as tf
from tensorflow import keras
import numpy as np

import os
from random import randint

# Parameters
dataset_dir = os.path.join('dataset', 'data') 
image_height = 28
image_width = 28
batch_size = 32
no_classes = 26
shuffle_seed = randint(111111111, 999999999)

# Loading dataset
train_dataset = keras.preprocessing.image_dataset_from_directory(
   dataset_dir,
   label_mode = 'categorical',
   color_mode = 'grayscale',
   image_size = (image_height, image_width),
   shuffle = True,
   seed = shuffle_seed,
   subset = 'training',
   validation_split = 0.2,
)

test_dataset = keras.preprocessing.image_dataset_from_directory(
   dataset_dir,
   label_mode = 'categorical',
   color_mode = 'grayscale',
   image_size = (image_height, image_width),
   shuffle = True,
   seed = shuffle_seed,
   subset = 'validation',
   validation_split = 0.2,
)

# Optimizing dataset
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size = AUTOTUNE)

normalization_layer = keras.layers.experimental.preprocessing.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(train_dataset))
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(test_dataset))

flattening_layer = keras.layers.Flatten()
train_dataset = train_dataset.map(lambda x, y: (flattening_layer(x), y))
image_batch, labels_batch = next(iter(train_dataset))
test_dataset = test_dataset.map(lambda x, y: (flattening_layer(x), y))
image_batch, labels_batch = next(iter(test_dataset))

# MLP Model Defining
model = keras.Sequential([
   keras.layers.Dense(units = 784, input_shape=(image_height*image_width,), activation = 'sigmoid'),
   keras.layers.Dense(units = 406, activation = 'sigmoid'),
   keras.layers.Dense(units = no_classes, activation='linear')
])

# Model Compiling
model.compile(
   optimizer = 'adam',
   loss = keras.losses.CategoricalCrossentropy(from_logits = True),
   metrics = ['accuracy']
)

# Predicting
history = model.fit(
  train_dataset,
  validation_data=test_dataset,
  epochs=50
)

prediction = model.predict_classes(test_dataset, verbose = 1)
print(prediction)

for test_entry in test_dataset.take(1):
   print(test_entry)