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
#data['url'].values

#convert it into numpy array and shuffle the dataset
#data = np.array(data)

#y = [d[1] for d in data]
#corpus = [d[0] for d in data]
#vectorizer = TfidfVectorizer(tokenizer=getTokens)
#X = vectorizer.fit_transform(corpus)


from sklearn.utils import shuffle

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
