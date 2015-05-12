
from __future__ import print_function
import numpy as np
import googleprediction as gp
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


model = gp.GooglePredictor(
    "vital-form-93916",
    "mysbucket/final_train.csv",
    "My First Project",
    "client_secrets.json")
    
model.list()

model.fit('CLASSIFICATION')

model.get_params()

X_test = np.genfromtxt('val_test.csv', delimiter=',', skip_header=1)

result = model.predict(X_test)

