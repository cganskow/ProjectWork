



from sklearn.utils import shuffle

data = pd.read_csv("/workspaces/Malicious-URL-Detection-using-Machine-Learning/data/dataNN.csv",on_bad_lines='skip')

data = shuffle(data)
url_train = data['url'][:].values
label_train = data['label'][:].values
url_test = data['url'][:].values
label_test = data['label'][:].values


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import time
import matplotlib.pyplot as plt
from codecarbon import EmissionsTracker
import logging

tracker = EmissionsTracker(allow_multiple_runs=True)

pipeline = Pipeline([
        ("vectorizer", TfidfVectorizer(tokenizer=None)),
        ("classifier", LogisticRegression())])

times = []
accuracy = []
carbon = []

train_sizes = np.arange(.1, 1.1, .1)

for size in train_sizes:
    logging.getLogger("codecarbon").setLevel(logging.CRITICAL)
    trainers = int(url_train.shape[0] * size)

    X_trainer = url_train[:trainers]
    y_trainer = label_train[:trainers]

    tracker.start()
    start = time.time()

    pipeline.fit(X_trainer, y_trainer)

    end = time.time()
    emissions: float = tracker.stop()

    accuracy_of_section = pipeline.score(url_test, label_test)

    times.append(end-start)
    accuracy.append(accuracy_of_section)
    carbon.append(emissions)

    print(f"Size: {size}\t | Time: {end-start}\t | Accuracy: {accuracy_of_section}\t | Carbon: {emissions} kg CO2")

plt.title("Pipeline")
plt.xlabel("Time (seconds)")
plt.ylabel("Accuracy")
plt.scatter(times, accuracy)
plt.show()

plt.title("Pipeline")
plt.xlabel("Time (seconds)")
plt.ylabel("CO2 Emissions (kg)")
plt.scatter(times, carbon)
plt.show()

plt.tight_layout()

del tracker
