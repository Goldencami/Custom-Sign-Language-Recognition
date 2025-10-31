import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

DATA_DIR = './data'

if len(os.listdir(DATA_DIR)) != 3:
    print('Error: No data found.')
    exit()

data = [] # ['sleepy.csv', 'thank-you.csv', 'hug.csv']
labels = []

for file in os.listdir(DATA_DIR):
    gesture = file.split('.')[0] # we only want the name
    data_dir = os.path.join(DATA_DIR, file)
    df = pd.read_csv(data_dir, header=0)
    data.append(df.values)
    # there are many landmarks per gestures, classify them
    labels.extend([gesture] * len(df))

# Combine all gesture arrays
X = np.vstack(data)
y = np.array(labels)

# stratify -> keep same proportion of all our different labels (1/3 each)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_predict = model.predict(X_test)

score = accuracy_score(y_test, y_predict) # probability, so 0 to 1
print('{}% of samples were classified correctly!'.format(score*100))

# save model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)