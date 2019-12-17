from process import *
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow import keras
from keras import regularizers
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.datasets.cifar import load_batch

source_file = "spreadspoke_scores.csv"
output_file = "results.txt"

def main():
    #initialize test and training data
    train_data,test_data = read_csv(source_file, 0)
    train_data = np.array(train_data)
    #shuffle training data
    np.random.shuffle(train_data)
    #split features and labels
    test_data = np.array(test_data)
    y_train = train_data[:,-1]
    X_train = train_data
    X_train = np.delete(X_train, -1, axis=1)
    y_test = test_data[:,-1]
    X_test = test_data
    X_test = np.delete(X_test, -1, axis=1)
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int32)

    #normalize data
    mean_pixel = X_train.mean(keepdims=True)
    std_pixel = X_train.std(keepdims=True)
    X_train = (X_train - mean_pixel) / std_pixel
    X_test = (X_test - mean_pixel) / std_pixel

    #train model
    epochs = 20
    n,p = X_train.shape
    print(p)
    model = get_compiled_model(p, keras.optimizers.Adam())
    history = train(X_train, y_train, epochs, model)
    #print(history.history)

    #test model
    results = test(X_test, y_test, model)
    print(results)
    print(get_bookie_score(source_file))
    #print(results)

    #print(X_train[0])



def train(X, y, epochs, model, validation_split = .1):
        history = model.fit(X, y, batch_size=30, epochs=epochs, validation_split = validation_split)
        return history

def test(x_test, y_test, model, batch_size = 1):
    results = model.evaluate(x_test, y_test, batch_size)
    return results

def get_uncompiled_model(p):
    inputs = tf.keras.Input(shape = (p,), name = "input")
    #x = Dense(64,
        #kernel_regularizer=regularizers.l2(0.01),
        #activity_regularizer=regularizers.l1(0.01))(inputs)
    x = Dense(100, tf.nn.relu)(inputs)
    x = Dense(10, tf.nn.relu)(x)
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

def print_epoch_scores(history):
    index = -1
    score = 0
    i = 0
    for acc in history.history.get("val_sparse_categorical_accuracy"):
        if acc > score:
            score = acc
            index = i
        i += 1

    i = 0
    for acc in history.history.get("val_sparse_categorical_accuracy"):
        if i == index:
            print(str(i) + ': ' + str(acc) + ' (PEAK)')
        else:
            print(str(i) + ': ' + str(acc))
        i += 1

def write_output(output):
    outString = ""
    for i in range(len(output)):
        outString += output[i] + "\n"
    f = open(output_file,"w+")
    f.write(outString)
    f.close()

if __name__ == "__main__" :
    main()
