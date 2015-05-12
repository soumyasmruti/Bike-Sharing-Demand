
from __future__ import print_function
import numpy as np
import googleprediction as gp

model = gp.GooglePredictor(
    "vital-form-93916",
    "mysbucket/final_train.csv",
    "My First Project",
    "client_secrets.json")
    
model.list()

model.fit('CLASSIFICATION')

model.get_params()

X_train = np.genfromtxt('final_test.csv', delimiter=',', skip_header=1)

out = model.predict(X_train)