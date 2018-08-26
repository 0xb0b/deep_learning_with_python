from keras.datasets import reuters
from keras.utils import to_categorical
from keras import models, layers, optimizers, losses, metrics
import numpy
import matplotlib.pyplot as viz


max_index = 10000

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(
    num_words=max_index)


# data preparation - 1-hot encoding in this case
# only the information about the bag of words is encoded
# the words order, sentence partitioning etc is discarded
# so the sample is encoded as a sum of 1-hot vectors, for each word in a text
def vectorize(sequences, dim):
    encoded = numpy.zeros((len(sequences), dim))
    for i, seq in enumerate(sequences):
        encoded[i, seq] += 1.0
    return encoded


x_train = vectorize(train_data, max_index)
x_test = vectorize(test_data, max_index)

y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)


# build the network - sequence of layers
# hyperparameters:
#     number of the hidden layers
#     number of units in each hidden layer
#     learning rate
num_units = [256, 128, 64]
num_hlayers = len(num_units)
learning_rate = 0.0001

model = models.Sequential()
# first hidden layer with input shape specified
model.add(layers.Dense(num_units[0], activation='relu',
                       input_shape=(max_index,)))
for i in range(1, num_hlayers):
    model.add(layers.Dense(num_units[i], activation='relu'))
# output layer
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(lr=learning_rate),
              loss=losses.categorical_crossentropy,
              metrics=[metrics.categorical_accuracy])


# train the model
# hyperparameters:
#     batch size
#     number of epochs
num_epochs = 40
history = model.fit(x_train, y_train, batch_size=512, epochs=num_epochs,
                    validation_split=0.2).history


# vizualization
epochs = range(1, num_epochs + 1)

viz.plot(epochs, history['loss'], 'bo', label='training loss')
viz.plot(epochs, history['val_loss'], 'b+', label='validation loss')
viz.title('training and validation loss')
viz.xlabel('epochs')
viz.ylabel('loss')
viz.legend()
viz.show()

viz.plot(epochs, history['categorical_accuracy'], 'ro', label='training accuracy')
viz.plot(epochs, history['val_categorical_accuracy'], 'r+',
         label='validation accuracy')
viz.title('training and validation accuracy')
viz.xlabel('epochs')
viz.ylabel('accuracy')
viz.legend()
viz.show()


# evaluate
_, test_accuracy = model.evaluate(x_test, y_test)
print('test data accuracy: {}'.format(test_accuracy))


# [64, 64]
#   starts overfitting after epoch 3: 0.96 training vs 0.78 test accuracy
# [64, 4] - bottleneck layer
#   starts overfitting after epoch 10: 0.9 training vs 0.7 test accuracy
#   drop of performance due to bottleneck
# [256, 256]
#   same as [64, 64]
# [256, 128, 64]
#   same as [64, 64]
#   model is less stable with more layers and units
#   learning rate 0.0001
#     much more stable
#     slowly approaches the performance values of 0.001 rate
#     starts overfitting later - after epoch 10
