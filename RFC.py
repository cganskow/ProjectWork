




from sklearn.ensemble import RandomForestClassifier
import time
import matplotlib.pyplot as plt
from codecarbon import EmissionsTracker
import logging

tracker = EmissionsTracker(allow_multiple_runs=True)
m = RandomForestClassifier(n_estimators=100)


times = []
accuracy = []
carbon = []

train_sizes = np.arange(.1, 1.1, .1)

for size in train_sizes:
    logging.getLogger("codecarbon").setLevel(logging.CRITICAL)
    trainers = int(X_train.shape[0] * size)

    X_trainer = X_train[:trainers]
    y_trainer = y_train[:trainers]

    tracker.start()
    start = time.time()
    
    m.fit(X_trainer, y_trainer)

    end = time.time()
    emissions: float = tracker.stop()

    accuracy_of_section = m.score(X_test, y_test)

    times.append(end-start)
    accuracy.append(accuracy_of_section)
    carbon.append(emissions)

    print(f"Size: {size}\t | Time: {end-start}\t | Accuracy: {accuracy_of_section}\t | Carbon: {emissions} kg CO2")

plt.title("Random Forest Classifier")
plt.xlabel("Times (seconds)")
plt.ylabel("Accuracy")
plt.scatter(times, accuracy)
plt.show()

plt.title("Random Forest Classifier")
plt.xlabel("Time (seconds)")
plt.ylabel("CO2 Emissions (kg)")
plt.scatter(times, carbon)
plt.show()

plt.tight_layout()

del tracker
