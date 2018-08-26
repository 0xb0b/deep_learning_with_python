from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models, layers, optimizers, losses, metrics
import matplotlib.pyplot as viz


(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

# data preparation
num_train_images, width, height = train_data.shape
num_pixels = width * height
x_train = train_data.reshape(num_train_images,
                             num_pixels).astype('float32') / 255

num_test_images, _, _ = test_data.shape
x_test = test_data.reshape(num_test_images,
                           num_pixels).astype('float32') / 255

y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)


# build the network
# hyperparameters:
#     architecture: number of hidden layers, number of units in each hidden
#                   layer
#     optimization: learning rate
num_units = [512]
num_hidden_layers = len(num_units)
learning_rate = 0.0001

model = models.Sequential()
# first hidden layer with input shape specified
model.add(layers.Dense(num_units[0], activation='relu',
                       input_shape=(num_pixels,)))
for i in range(1, num_hidden_layers):
    model.add(layers.Dense(num_units[i], activation='relu'))
# output layer
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(lr=learning_rate),
              loss=losses.categorical_crossentropy,
              metrics=[metrics.categorical_accuracy])


# train the model
# hyperparameters:
#   batch size
#   number of epochs
num_epochs = 20
history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=128,
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


# changing hyperparameters has little effect on the model performance
# layers: [512] vs [2048] vs [512, 256, 64, 32]
# number of epochs: 3, 5, 8, 20
# learning rate 0.001
#   --> overfitting 0.999 training vs 0.98 test accuracy, starts after epoch 3
# [512] and learning rate 0.01
#   --> lower overfitting 0.986 training vs 0.973 test accuracy
# [512] and learning rate 0.0001
#   --> lower overfitting 0.984 training vs 0.975 test accuracy,
#       starts after epoch 7

# almost no dependence of performance on hyperparameters
#   what is the simplest network with a similar performance for MNIST dataset?
#     use Bayesian optimization to find hyperparameters values
#     explore the dynamic architecture - graph instead of layer