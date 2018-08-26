from keras.datasets import imdb
from keras import models, preprocessing
import numpy
import imdb_samples

word_index = imdb.get_word_index()

max_index = 10000
start_char = 1
index_from = 3
x = numpy.zeros((len(imdb_samples.reviews), max_index))
labels = []
for k, (text, label) in enumerate(imdb_samples.reviews):
    labels.append(label)
    sequence = [start_char]
    words = preprocessing.text.text_to_word_sequence(text)
    for w in words:
        i = word_index.get(w, 0) + index_from
        if index_from < i < max_index:
            sequence.append(i)
    x[k, sequence] += 1.0

model = models.load_model('3_4_classifying_movie_reviews_model.h5')
predictions = model.predict(x)
print('new imdb reviews predictions')
num_correct = 0
for i, p in enumerate(predictions):
    if abs(p[0] - labels[i]) < 0.5:
        num_correct += 1
    print(p, labels[i])
accuracy = num_correct / len(labels)
print('accuracy: {}'.format(accuracy))
