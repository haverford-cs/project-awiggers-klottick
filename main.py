from process import *
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
    epochs = 1
    n,p = X.shape
    train_dset = tf.data.Dataset.from_tensor_slices((X, y))
    train_dset = train_dset.shuffle(X.shape[0])

    model = get_compiled_model(p, keras.optimizers.Adam())
    history = train(X, y, epochs, model)
    print(history.history)

def train(X, y, epochs, model, validation_split = .1):
        history = model.fit(X, y, batch_size=1, epochs=epochs, validation_split = 0.1)
        return history
def get_uncompiled_model(p):
    inputs = tf.keras.Input(shape = (p,), name = "input")
    x = Dense(10, tf.nn.relu)(inputs)
    outputs = Dense(3, tf.nn.softmax)(x)
    model = tf.keras.Model(inputs = inputs, outputs = outputs)
    return model
def get_compiled_model(p,optimizer):
    model = get_uncompiled_model(p)
    model.compile(optimizer=optimizer,  # Optimizer
              # Loss function to minimize
              loss=keras.losses.SparseCategoricalCrossentropy(),
              # List of metrics to monitor
              metrics=[keras.metrics.SparseCategoricalAccuracy()])
    return model
if __name__ == "__main__" :
    main()
