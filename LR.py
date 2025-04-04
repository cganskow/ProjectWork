import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import re
import random
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def getTokens(input):
    tokensBySlash = str(input.encode('utf-8')).split('/')
#    print(f".encode(utf-8).split('/'){tokensBySlash}")
    allTokens = []
    for i in tokensBySlash:
        tokens = str(i).split('-')
        tokensByDot = []
        for j in range(0,len(tokens)):
            tempTokens = str(tokens[j]).split('.')
            tokensByDot = tokensByDot + tempTokens
        allTokens = allTokens + tokens + tokensByDot
    allTokens = list(set(allTokens))
    if 'com' in allTokens:
        allTokens.remove('com')
    return allTokens

#function to remove "http://" from URL
def trim(url):
    return re.match(r'(?:\w*://)?(?:.*\.)?([a-zA-Z-1-9]*\.[a-zA-Z]{1,}).*', url).groups()[0]

data = pd.read_csv("dataNN.csv",on_bad_lines='skip')	#reading file
# data = pd.read_csv("/workspaces/Malicious-URL-Detection-using-Machine-Learning/data/dataNN.csv",on_bad_lines='skip')	#reading file
data['url'].values

#convert it into numpy array and shuffle the dataset
data = np.array(data)
random.shuffle(data)

y = [d[1] for d in data]
corpus = [d[0] for d in data]
vectorizer = TfidfVectorizer(tokenizer=getTokens)
X = vectorizer.fit_transform(corpus)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#1 - Logistic Regression
import time
import matplotlib.pyplot as plt
from codecarbon import EmissionsTracker
import logging

tracker = EmissionsTracker(allow_multiple_runs=True)
model = LogisticRegression(C=1)

times = []
accuracy = []
carbon = []

train_sizes = np.arange(0.1, 1.1, 0.1)

print("\n")

for size in train_sizes:
    logging.getLogger("codecarbon").setLevel(logging.CRITICAL)
    trainers = int(X_train.shape[0] * size)

    X_trainer = X_train[:trainers]
    y_trainer = y_train[:trainers]

    tracker.start()
    start = time.time()

    model.fit(X_trainer, y_trainer)

    end = time.time()
    emissions: float = tracker.stop()

    y_predict = model.predict(X_test)

    accuracy_of_section = accuracy_score(y_test, y_predict)

    times.append(end-start)
    accuracy.append(accuracy_of_section)
    carbon.append(emissions)
    print(f"Size: {size}\t | Time: {end-start}\t | Accuracy: {accuracy_of_section}\t | Carbon: {emissions} kg CO2")


plt.title("Logistic Regression")
plt.xlabel("Time (seconds)")
plt.ylabel("Accuracy")
plt.scatter(times, accuracy)
plt.show()

plt.title("Logistic Regression")
plt.xlabel("Time (seconds)")
plt.ylabel("CO2 Emissions (kg)")
plt.scatter(times, carbon)
plt.show()

plt.tight_layout()

del tracker
