# Import TensorFlow

# import tensorflow as tf

# # Check its version

# tf.__version__

# # Train a FeedForward Neural Network for Image Classification

# import numpy as np

# print('Loading...\n')
# data = np.loadtxt('data/mnist.csv', delimiter=',')
# print('Data has been loaded.\n')

# X_train = data[:,1:]
# Y_train = data[:,0]
# X_train = X_train / 255.

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(16, activation='relu'),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.fit(X_train, Y_train, epochs=5, batch_size=32)

# print('Model Training Completed\n')

import tensorflow as tf
print(tf.reduce_sum(tf.random_normal([1000, 1000])))