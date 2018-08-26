from keras.datasets import imdb
from keras import models, layers, optimizers, losses, metrics
import numpy
import matplotlib.pyplot as viz


# input data - training and test sets

max_index = 10000

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words=max_index)


# data preparation - 1-hot encoding in this case
# only the information about the set of words is encoded
# the word frequencies, order, sentence partitioning etc is discarded
# so the sample is encoded as a sum of 1-hot vectors, for each word in a words
# set of the review

def encode_1hot(sequences, dim):
    encoded = numpy.zeros((len(sequences), dim))
    for i, seq in enumerate(sequences):
        encoded[i, seq] = 1.0
    return encoded


# only the information about the bag of words is encoded
# the words order, sentence partitioning etc is discarded
# so the sample is encoded as a sum of 1-hot vectors, for each word in a review
def vectorize(sequences, dim):
    encoded = numpy.zeros((len(sequences), dim))
    for i, seq in enumerate(sequences):
        encoded[i, seq] += 1.0
    return encoded


x_train = vectorize(train_data, max_index)
x_test = vectorize(test_data, max_index)

y_train = numpy.asarray(train_labels).astype('float32')
y_test = numpy.asarray(test_labels).astype('float32')


# build the network - sequence of layers
# hyperparameters:
#     number of the hidden layers
#     number of units in each hidden layer
#     learning rate
num_hlayers = 16
num_units = [256] * num_hlayers
learning_rate = 0.001

model = models.Sequential()
# first hidden layer with input shape specified
model.add(layers.Dense(num_units[0], activation='relu',
                       input_shape=(max_index,)))
for i in range(1, num_hlayers):
    model.add(layers.Dense(num_units[i], activation='relu'))
# output layer
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=learning_rate),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])


# train the model
# hyperparameters:
#     batch size
#     number of epochs
num_epochs = 16
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

viz.plot(epochs, history['binary_accuracy'], 'ro', label='training accuracy')
viz.plot(epochs, history['val_binary_accuracy'], 'r+',
         label='validation accuracy')
viz.title('training and validation accuracy')
viz.xlabel('epochs')
viz.ylabel('accuracy')
viz.legend()
viz.show()

# models.save_model(model, "3_4_classifying_movie_reviews_model.h5")

predictions = model.predict(x_test)
num_correct = 0
for i, p in enumerate(predictions):
    if abs(p[0] - y_test[i]) < 0.5:
        num_correct += 1
accuracy = num_correct / len(y_test)
print('accuracy: {}'.format(accuracy))



# open questions:
# hyperparameters tuning
# change of architecture
