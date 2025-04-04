




#2 - SVM
import time
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from codecarbon import EmissionsTracker
import logging

tracker = EmissionsTracker(allow_multiple_runs=True)
svcModel = SVC()

times = []
accuracy = []
carbon=[]

train_sizes = np.arange(.1, 1.1, .1)

for size in train_sizes:
    logging.getLogger("codecarbon").setLevel(logging.CRITICAL)
    trainers = int(X_train.shape[0] * size)

    X_trainer = X_train[:trainers]
    y_trainer = y_train[:trainers]
    
    tracker.start()
    start = time.time()

    svcModel.fit(X_trainer, y_trainer)

    end = time.time()
    emissions: float = tracker.stop()

    y_predict = svcModel.predict(X_test)
    accuracy_of_section = accuracy_score(y_test, y_predict)

    times.append(end-start)
    accuracy.append(accuracy_of_section)
    carbon.append(emissions)

    print(f"Size: {size}\t | Time: {end-start}\t | Accuracy: {accuracy_of_section}\t | Carbon: {emissions} kg CO2")

plt.title("SVC Model")
plt.xlabel("Time (seconds)")
plt.ylabel("Accuracy")
plt.scatter(times, accuracy)
plt.show()

plt.title("SVC Model")
plt.xlabel("Time (seconds)")
plt.ylabel("CO2 Emissions (kg)")
plt.scatter(times, carbon)
plt.show()

plt.tight_layout()

del tracker
