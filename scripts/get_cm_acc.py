import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import glob

# Load model
model = tf.keras.models.load_model('../predictive_maintenance/model_v18.keras')
max_vals = np.array([1272.0, 3094.0, 65535.0])

all_labels = []
all_preds = []

files = sorted(glob.glob("../predictive_maintenance/Dataset_*.csv"))
for f in files:
    df = pd.read_csv(f)
    X = df[['MinDistance', 'ApproachVelocity', 'TTC']].values
    y = df['Label'].values
    
    X_scaled = X / max_vals
    preds = np.argmax(model.predict(X_scaled, batch_size=4096, verbose=0), axis=1)
    
    all_labels.extend(y)
    all_preds.extend(preds)

cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
total = np.sum(cm)
correct = np.trace(cm)
acc = correct / total * 100

print(f"Total Samples: {total}")
print(f"Correct Predictions (Trace): {correct}")
print(f"Overall Accuracy (Micro-average): {acc:.2f}%")
print("Confusion Matrix:")
print(cm)
