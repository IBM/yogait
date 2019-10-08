import sys
import pickle
import numpy as np
import json

from utils.helpers import convert

with open('./../assets/classifier.pkl', 'rb') as fil:
    classifier = pickle.load(fil)

with open('./../assets/classes.pkl', 'rb') as fil:
    name_map = pickle.load(fil)

# set up inference data correctly
preds = json.loads(sys.argv[1].split('\n')[3])
coordinates = np.array([[d['x'], d['y']] for d in preds], dtype=np.float32)
frame = convert(coordinates)
missing_vals = 19 - frame.shape[0]
frame = np.concatenate([frame, np.zeros([missing_vals, 2])])
nx, ny = frame.shape
reshaped_frame = frame.reshape((1, nx*ny))

# make predictions
prediction = classifier.predict(reshaped_frame)
confidence = classifier.predict_proba(reshaped_frame)
current_prediction = name_map[prediction[0]]
score = 100*confidence[0][prediction[0]]

# return result to stdout, where node.js is listening
response = str(current_prediction)+","+str(score)
print(response)
