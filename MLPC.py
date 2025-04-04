




from sklearn.utils import shuffle

data = pd.read_csv("/workspaces/Malicious-URL-Detection-using-Machine-Learning/data/dataNN.csv",on_bad_lines='skip')

data = shuffle(data)
url_train = data['url'][:].values
label_train = data['label'][:].values
url_test = data['url'][:].values
label_test = data['label'][:].values

# MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
import time
import matplotlib.pyplot as plt
import numpy as np
from codecarbon import EmissionsTracker
import logging
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

tracker = EmissionsTracker(allow_multiple_runs=True)
vectorizer = TfidfVectorizer()
clf = MLPClassifier()

times = []
accuracy = []
carbon = []

train_sizes = np.arange(.1, 1.1, .1)

X_train_ft = vectorizer.fit_transform(url_train)
X_test_t = vectorizer.transform(url_test)

for size in train_sizes:
    logging.getLogger("codecarbon").setLevel(logging.CRITICAL)
    trainers = int(url_train.shape[0] * size)

    X_trainer = X_train_ft[:trainers]
    y_trainer = le.fit_transform(label_train[:trainers])

    tracker.start()
    start = time.time()

    clf.fit(X_trainer, y_trainer)

    end = time.time()
    emissions: float = tracker.stop()

    accuracy_of_section = clf.score(X_test_t, label_test)

    times.append(end-start)
    accuracy.append(accuracy_of_section)
    carbon.append(emissions)

    print(f"Size: {size}\t | Time: {end-start}\t | Accuracy: {accuracy_of_section}\t | Carbon: {emissions} kg CO2")




plt.title("MLPClassifier")
plt.xlabel("Time (seconds)")
plt.ylabel("Accuracy")
plt.scatter(times, accuracy)
plt.show()

plt.title("MLPClassifier")
plt.xlabel("Time (seconds)")
plt.ylabel("CO2 Emissions (kg)")
plt.scatter(times, carbon)
plt.show()

plt.tight_layout()

del tracker
