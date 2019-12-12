from process import *
import fc_nn
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.datasets.cifar import load_batch

source_file = "spreadspoke_scores.csv"

def main():
    data = np.array(read_csv(source_file))
    y = data[:,-1]
    X = data
    X = np.delete(X, -1, axis=1)


    train_dset = tf.data.Dataset.from_tensor_slices((X, y))
    train_dset = train_dset.shuffle(X.shape[0])

    inputs = tf.keras.Input(shape = (92,), name = "input")
    x = Dense(10, tf.nn.relu)(inputs)
    outputs = Dense(3, tf.nn.softmax)(x)
    model = tf.keras.Model(inputs = inputs, outputs = outputs)

    model.compile(optimizer=keras.optimizers.Adam(),  # Optimizer
              # Loss function to minimize
              loss=keras.losses.SparseCategoricalCrossentropy(),
              # List of metrics to monitor
              metrics=[keras.metrics.SparseCategoricalAccuracy()])

    history = model.fit(X, y, batch_size=1, epochs=10, validation_split = 0.1)

    print(history.history)

if __name__ == "__main__" :
    main()
