"""
Train and test a fully-connected neural network on football data.
Author: Alton Wiggers, Kadan Lottick
Date: 12/17/19
"""

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
    year = 0
    epochs = 10
    batch_size = 50
    single_test(year, epochs, batch_size)
    #multi_test(epochs, batch_size)




def multi_test(epochs, batch_size):
    """
    Create multiple models using the given epochs and
    batch_size. Runs several times with datasets of games
    at and after benchmark years, averaging the results of
    models for a given year. Then prints results to an output
    file.
    """
    years = [1979, 1985, 1990, 1995, 2000, 2005, 2010, 2012]
    count = 20
    output = []
    output.append("year,test: train acc, test acc")
    for year in years:
        train_scores = []
        test_scores = []

        for i in range(count):
            history,results = runModel(year, epochs, batch_size)
            s = str(year) + " test " + str(i) + ": "
            s += str(results[0]) + ", " + str(results[1])
            output.append(s)
            train_scores.append(results[0])
            test_scores.append(results[1])

        train_sum = 0
        test_sum = 0
        for i in range(count):
            train_sum += train_scores[i]
            test_sum += test_scores[i]
        train_avg = train_sum / count
        test_avg = test_sum / count
        s = str(year) + " averages " + str(i) + ": "
        s += str(train_avg) + ", " + str(test_avg)
        output.append(s)

    write_output(output)


def single_test(year, epochs, batch_size):
    """
    Create a single model with dataset of given year
    using given epochs and batch_size. Then print results
    for the model.
    """
    history,results = runModel(year, epochs, batch_size)

    #print(history.history)
    print_epoch_scores(history)
    print(results)
    print(get_bookie_score(source_file))


def runModel(startYear, epochs, batch_size):
    """
    Generate a dataset using the given startYear.
    Split into training, testing, and validation data
    and normalize the dataset. Then train and test
    a fully connected neural network.
    """
    #initialize test and training data
    train_data,test_data = read_csv(source_file, startYear)
    train_data = np.array(train_data)
    test_data = np.array(test_data)

    #shuffle training data
    np.random.shuffle(train_data)
    np.random.shuffle(test_data)
    #test_data = np.delete(test_data, -1, axis=0)
    val_data = np.split(test_data, 2)
    test_data = val_data[0]
    val_data = val_data[1]

    #split features and labels
    test_data = np.array(test_data)
    y_train = train_data[:,-1]
    X_train = train_data
    X_train = np.delete(X_train, -1, axis=1)
    y_test = test_data[:,-1]
    X_test = test_data
    X_test = np.delete(X_test, -1, axis=1)
    y_val = val_data[:,-1]
    X_val = val_data
    X_val = np.delete(X_val, -1, axis=1)
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.int32)

    #normalize data
    mean_pixel = X_train.mean(keepdims=True)
    std_pixel = X_train.std(keepdims=True)
    X_train = (X_train - mean_pixel) / std_pixel
    X_test = (X_test - mean_pixel) / std_pixel
    X_val = (X_val - mean_pixel) / std_pixel

    #train model
    n,p = X_train.shape
    model = get_compiled_model(p, keras.optimizers.Adam())
    history = train(X_train, y_train, epochs, model, (X_val, y_val), batch_size)


    #test model
    results = test(X_test, y_test, model)

    return history,results





def train(X, y, epochs, model, val_data, batch_size):
    """
    fit the given training data (X,y) to given model.
    Use given validation data and batch size.
    """
        history = model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_data = val_data)
        return history

def test(x_test, y_test, model, batch_size = 1):
    """
    evaluate the given test data (X,y) to given model.
    """
    results = model.evaluate(x_test, y_test, batch_size)
    return results

def get_uncompiled_model(p):
    """
    Compiles a fully-connected neural network; the architecture is:
    fully-connected (dense) layer -> ReLU -> ReLU -> fully connected layer.
    """
    inputs = tf.keras.Input(shape = (p,), name = "input")
    #x = Dense(64,
        #kernel_regularizer=regularizers.l2(0.01),
        #activity_regularizer=regularizers.l1(0.01))(inputs)
    x = Dense(100, tf.nn.relu)(inputs)
    x = Dense(10, tf.nn.relu)(x)
    outputs = Dense(2, tf.nn.softmax)(x)
    model = tf.keras.Model(inputs = inputs, outputs = outputs)
    return model

def get_compiled_model(p,optimizer):
    """
    Generates a model with given feature size
    and desired optimizer.
    """
    model = get_uncompiled_model(p)
    model.compile(optimizer=optimizer,  # Optimizer
              # Loss function to minimize
              loss=keras.losses.SparseCategoricalCrossentropy(),
              # List of metrics to monitor
              metrics=[keras.metrics.SparseCategoricalAccuracy()])
    return model

def print_epoch_scores(history):
    """
    Prints out validation accuracy for each epoch
    of training. Marks the highest score.
    """
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
    """
    Writes data to a file.
    """
    outString = ""
    for i in range(len(output)):
        outString += output[i] + "\n"
    f = open(output_file,"w+")
    f.write(outString)
    f.close()

if __name__ == "__main__" :
    main()
