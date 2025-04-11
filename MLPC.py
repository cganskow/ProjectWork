import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.metrics import accuracy_score
import time
from codecarbon import EmissionsTracker
import logging
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

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

from sklearn.utils import shuffle

data = shuffle(data)
url_train = data['url'][:].values
label_train = data['label'][:].values
url_test = data['url'][:].values
label_test = data['label'][:].values

# MLPClassifier
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=int(100)):
        super(MLP,self).__init__()

        assert isinstance(input_dim, int), f"Expected input_dim to be an integer, got {type(input_dim)}"
        assert isinstance(hidden_dim, int), f"Expected hidden_dim to be an integer, got {type(hidden_dim)}"
        
        # Make sure input_dim is not zero or negative
        assert input_dim > 0, f"input_dim should be greater than 0, got {input_dim}"

        self.model = nn.Sequential (
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
    def forward(self, x):
        return self.model(x)

tracker = EmissionsTracker(allow_multiple_runs=True)
vectorizer = TfidfVectorizer(max_features=50000)

times = []
accuracy = []
carbon = []

train_sizes = np.arange(.1, 1.1, .1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train_ft = vectorizer.fit_transform(url_train)
X_test_t = vectorizer.transform(url_test)
X_train_np = X_train_ft.astype(np.float32)
X_test_np = X_test_t.astype(np.float32)
print("X_train_np ", X_train_np.shape)
print("X_train_ft ", X_train_ft.shape)

label_train_encoded = le.fit_transform(label_train)
label_test_encoded = le.transform(label_test)
X_test_tensor = torch.tensor(X_test_np.toarray(), dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(label_test_encoded).long().to(device)

for size in train_sizes:
    logging.getLogger("codecarbon").setLevel(logging.CRITICAL)
    trainers = int(X_train_np.shape[1] * size)
    X_slice = X_train_ft[:trainers]         
    X_dense = X_slice.toarray()             
    X_trainer = torch.tensor(X_dense, dtype=torch.float32).to(device)
    y_trainer = torch.tensor(label_train_encoded[:trainers]).long().to(device)

    train_dataset = TensorDataset(X_trainer, y_trainer)
    # batchsize?
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    clf = MLP(input_dim=X_trainer.shape[1]).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(clf.parameters(), lr=0.001)

    tracker.start()
    start = time.time()

    clf.train()
    num_epochs = 5
    for epoch in range(num_epochs):
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = clf(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    end = time.time()
    emissions: float = tracker.stop()

    clf.eval()
    with torch.no_grad():
        test_preds = clf(X_test_tensor.to(device))
        predicted = torch.argmax(test_preds, dim=1).cpu()
        acc = accuracy_score(y_test_tensor, predicted)

    #accuracy_of_section = clf.score(X_test_t, label_test)

    times.append(end-start)
    accuracy.append(acc)
    carbon.append(emissions)

    print(f"Size: {size}\t | Time: {end-start}\t | Accuracy: {acc}\t | Carbon: {emissions} kg CO2")


del tracker
