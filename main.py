"""
Train and test a fully-connected neural network on football data.
Author: Alton Wiggers, Kadan Lottick
Date: 12/17/19
"""

from process import *
import matplotlib.pyplot as plt
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
    '''
    Runs one of three kinds of tests given
    Start year and epochs.
    WARNING: multi-year tests take a long
    time to run.
    '''
    year = 0
    epochs = 50
    single_test(year, epochs)
    #multi_test_year(epochs)
    #multi_test_epochs(year)




def multi_test_year(epochs):
    """
    Create multiple models using the given epochs.
    Runs several times with datasets of games
    at and after benchmark years, averaging the results of
    models for a given year. Then prints results to an output
    file.
    """
    #list of start years
    years = [1979, 1985, 1990, 1995, 2000, 2005, 2010,2012]
    '''
    years = []
    for i in range(18):
        years.append(2*i + 1979)
    '''

    #number of models to generate for each start year
    count = 50
    output = []
    train_avgs = []
    test_avgs = []

    output.append("year,test: train acc, test acc")
    for year in years:
        train_scores = []
        test_scores = []

        #create a 'count' number of models
        for i in range(count):
            history,results,confusionMatrix = runModel(year, epochs)
            print(str(year) + " " + str(i))
            s = str(year) + " test " + str(i) + ": "
            s += str(results[0]) + ", " + str(results[1])
            output.append(s)
            #get training and testing scores
            print(results)
            print(history.history["sparse_categorical_accuracy"][-1])
            train_scores.append(history.history["sparse_categorical_accuracy"][-1])
            test_scores.append(results[1])

        train_sum = 0
        test_sum = 0
        #take averages for each year
        for i in range(count):
            train_sum += train_scores[i]
            test_sum += test_scores[i]
        train_avg = train_sum / count
        test_avg = test_sum / count
        train_avgs.append(train_avg)
        test_avgs.append(test_avg)
        s = str(year) + " averages " + str(i) + ": "
        s += str(train_avg) + ", " + str(test_avg)
        output.append(s)


    write_output(output)
    plotData(years, train_avgs, test_avgs, epochs)

def multi_test_epochs(year):
    """
    Create multiple models using the given start year
    Runs several times with datasets of games
    for different numbers of epochs, averaging the results of
    models for a given number of epochs. Then prints results to an output
    file.
    """
    #list of epochs
    epochs = [25,50,75,100,125]

    #number of models to generate for each epoch number
    count = 5
    output = []
    train_avgs = []
    test_avgs = []

    output.append("year,test: train acc, test acc")
    for epoch in epochs:
        train_scores = []
        test_scores = []

        #create a 'count' number of models
        for i in range(count):
            history,results,confusionMatrix = runModel(year, epoch)
            print(str(epoch) + " " + str(i))
            #get training and testing scores
            train_scores.append(history.history["sparse_categorical_accuracy"][-1])
            test_scores.append(results[1])

        train_sum = 0
        test_sum = 0
        #take averages for each epoch number
        for i in range(count):
            train_sum += train_scores[i]
            test_sum += test_scores[i]
        train_avg = train_sum / count
        test_avg = test_sum / count
        train_avgs.append(train_avg)
        test_avgs.append(test_avg)

    plotEpochData(epochs, train_avgs, test_avgs)


def single_test(year, epochs):
    """
    Create a single model with dataset of given year
    using given epochs. Then print results
    for the model.
    """
    history,results,confusionMatrix = runModel(year, epochs)

    print("testing loss/accuracy:")
    print(results)
    print("confusion matrix:")
    print(confusionMatrix)
    print("bookie scores:")
    print(get_bookie_score(source_file))


def runModel(startYear, epochs):
    """
    Generate a dataset using the given startYear.
    Split into training, testing, and validation data
    and normalize the dataset. Then train and test
    a fully connected neural network.
    """
    #initialize test and training data
    train_data,test_data,current_data = read_csv(source_file, startYear)
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    current_data = np.array(current_data)

    #shuffle training data
    np.random.shuffle(train_data)
    #np.random.shuffle(test_data)

    #split features and labels
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

    #used for running tests on games that have yet to happen
    y_cur = current_data[:,-1]
    X_cur = current_data
    X_cur = np.delete(X_cur, -1, axis=1)
    X_cur = np.asarray(X_cur, dtype=np.float32)
    y_cur = np.asarray(y_cur, dtype=np.int32)


    #normalize data
    mean_pixel = X_train.mean(keepdims=True)
    std_pixel = X_train.std(keepdims=True)
    X_train = (X_train - mean_pixel) / std_pixel
    X_test = (X_test - mean_pixel) / std_pixel
    X_cur = (X_cur - mean_pixel) / std_pixel

    #train model
    n,p = X_train.shape
    model = get_compiled_model(p, keras.optimizers.Adam())
    history = train(X_train, y_train, epochs, model, n)


    #test model
    results = test(X_test, y_test, model)

    confusionMatrix = generateConfusionMatrix(X_test, y_test, model)

    #print predictions for next week's games
    predictions = model.predict(X_cur)
    #print(predictions)

    keras.backend.clear_session()
    return history,results,confusionMatrix





def train(X, y, epochs, model, batch_size):
    """
    fit the given training data (X,y) to given model.
    Use given validation data and batch size.
    """
    history = model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split = 0)
    return history

def test(x_test, y_test, model, batch_size = 1):
    """
    evaluate the given test data (X,y) to given model.
    """
    results = model.evaluate(x_test, y_test, batch_size)
    return results

def get_uncompiled_model(p):
    """
    Creates a fully-connected neural network; the architecture is:
    fully-connected (dense) layer -> ReLU -> ReLU -> fully connected layer.
    """
    inputs = tf.keras.Input(shape = (p,), name = "input")
    x = Dense(500, tf.nn.relu)(inputs)
    outputs = Dense(2, tf.nn.softmax)(x)
    model = tf.keras.Model(inputs = inputs, outputs = outputs)
    return model

def get_compiled_model(p,optimizer):
    """
    Generates and compiles a model with given feature size
    and desired optimizer.
    """
    model = get_uncompiled_model(p)
    model.compile(optimizer=optimizer,  # Optimizer
              # Loss function to minimize
              loss=keras.losses.SparseCategoricalCrossentropy(),
              # List of metrics to monitor
              metrics=[keras.metrics.SparseCategoricalAccuracy()])

    return model

def generateConfusionMatrix(X_test, y_test, model):
    outcomes = model.predict(X_test)
    matrix = np.zeros((2,2))
    n,p = X_test.shape
    for i in range(n):
        prediction = 0
        if outcomes[i][1] > outcomes[i][0]:
            prediction = 1
        matrix[y_test[i]][prediction] +=1
    return matrix

def plotData(years, train_acc, test_acc, epochs) :
    """
    generate plot
    """
    plt.plot(years,train_acc, c='b', label = "Training Data")
    plt.plot(years,test_acc,c ='r', label = "Test Data")
    plt.title("Accuracy by Year for " + str(epochs) + " Epochs")
    plt.legend()
    plt.xlabel("Year", fontsize = 16)
    plt.ylabel("Accuracy", fontsize = 16)
    plt.show()

def plotEpochData(epochs, train_acc, test_acc) :
    """
    generate plot
    """
    plt.plot(epochs,train_acc, c='b', label = "Training Data")
    plt.plot(epochs,test_acc,c ='r', label = "Test Data")
    plt.title("Accuracy for Number of Epochs")
    plt.legend()
    plt.xlabel("Epochs", fontsize = 16)
    plt.ylabel("Accuracy", fontsize = 16)
    plt.show()


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
