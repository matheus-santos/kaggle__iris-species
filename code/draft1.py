# NN to classify Iris Specimens
# https://www.kaggle.com/uciml/iris

# TODOS:
# - Strategies to split into test and train datasets.
#       Maybe sklearn.model_selection.KFold suffices
# - Learn Cross-validation techniques for assessing how the results of a
#       statistical analysis will generalize to an independent data set
# - How to save a trained model?

from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
from pync import Notifier
import numpy
import sys

# Variables
classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

# Fix random seed for reproducibility
numpy.random.seed(7)

# 1. Load Iris species dataset
# Split into input (X) and output (Y) variables
Notifier.notify("Loading model", title="Iris")
dataset = numpy.loadtxt("Iris.formatted.csv", delimiter=",", skiprows=1)

training_idx = numpy.random.randint(dataset.shape[0], size=105)
test_idx = numpy.random.randint(dataset.shape[0], size=45)

X_train = dataset[training_idx, :4]
Y_train = to_categorical(dataset[training_idx, 4], num_classes=3)

X_test = dataset[test_idx, :4]
Y_test = to_categorical(dataset[test_idx, 4], num_classes=3)

# 2. Define Model
Notifier.notify("Defining network", title="Iris")
model = Sequential()  # Creating Sequential Model
model.add(Dense(32, input_dim=4, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(3, activation="sigmoid"))

# 3. Compile Model
Notifier.notify("Compiling model", title="Iris")
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(
    loss='categorical_crossentropy',
    optimizer=sgd,
    metrics=['accuracy']
)

# 4. Fit Model
Notifier.notify("Fitting model", title="Iris")
model.fit(
    X_train,
    Y_train,
    epochs=105,
    batch_size=10
)

# 5. Evaluate Model
Notifier.notify("Done! Evaluating model...", title="Iris")
scores = model.evaluate(X_test, Y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# 7. Make predictions
predictions = model.predict(X_test)
i = 0
for x in X_test:
    idx = predictions[i].argmax()
    eligible_class = classes[idx]
    print(numpy.array_str(x) + " is " + eligible_class)
    i += 1
