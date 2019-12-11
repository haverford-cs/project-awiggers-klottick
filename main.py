from process import *
from fc_nn import FCmodel
import numpy as np

source_file = "spreadspoke_scores.csv"

def main():
    data = np.array(read_csv(source_file))


def run_training(model, train_dset, val_dset):

    # set up a loss_object train_dset(sparse categorical cross entropy)
    # use the Adam optimizer
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    # set up metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy( \
        name='train_accuracy')

    # train for 10 epochs (passes over the data)
    train_acc = []


    for epoch in range(10):
        # TODO run training step
        for images, labels in train_dset:
            with tf.GradientTape() as t:
                predictions = model.call(images)
                loss = loss_object(labels, predictions)
                params = model.trainable_variables
            grads = t.gradient(loss, params)
            optimizer.apply_gradients(zip(grads,model.trainable_variables))
        # uncomment below
        train_loss(loss)
        train_accuracy(labels, predictions)

        template = 'Epoch {}, Loss: {}, Accuracy: {}'
        print(template.format(epoch+1,
                            train_loss.result(),
                            train_accuracy.result()*100)

        # Reset the metrics for the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
    return model,train_acc
