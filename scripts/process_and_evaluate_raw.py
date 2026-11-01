import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
import os

print("Loading V18 Model...")
model = tf.keras.models.load_model('../predictive_maintenance/model_v18.keras')
max_vals = np.array([1272.0, 3094.0, 65535.0])

def process_file(file_path):
    df = pd.read_csv(file_path)
    
    prev_valid_dist = 0
    prev_valid_timestamp_ms = 0
    smoothed_velocity_mmps = 0
    valid_sample_counter = 0
    initialized = False
    
    processed_rows = []
    
    for index, row in df.iterrows():
        # MinDistance selector
        dists = [row['TOF1_mm'], row['TOF2_mm'], row['TOF3_mm'], row['TOF4_mm']]
        valid_dists = [d for d in dists if d < 8190]
        
        min_dist = min(valid_dists) if valid_dists else 8190
        current_time_ms = int(row['Timestamp_s'] * 1000)
        
        velocity_mmps = 0
        
        if min_dist >= 8190:
            valid_sample_counter = 0
            continue
            
        if not initialized:
            initialized = True
            prev_valid_dist = min_dist
            prev_valid_timestamp_ms = current_time_ms
            smoothed_velocity_mmps = 0
            valid_sample_counter = 1
            velocity_mmps = 0
        else:
            if valid_sample_counter < 255:
                valid_sample_counter += 1
                
            if valid_sample_counter >= 2:
                dt_ms = current_time_ms - prev_valid_timestamp_ms
                if dt_ms >= 10:
                    dd = int(min_dist) - int(prev_valid_dist)
                    raw_vel = int(dd * 1000 / dt_ms)
                    
                    if valid_sample_counter == 2:
                        smoothed_velocity_mmps = raw_vel
                    else:
                        smoothed_velocity_mmps = smoothed_velocity_mmps + ((raw_vel - smoothed_velocity_mmps) // 4)
                        
                    abs_vel = abs(smoothed_velocity_mmps)
                    if min_dist < 400: deadband = 20
                    elif min_dist < 600: deadband = 55
                    elif min_dist < 800: deadband = 125
                    elif min_dist < 1000: deadband = 225
                    else: deadband = 400
                    
                    if abs_vel < deadband:
                        velocity_mmps = 0
                    else:
                        velocity_mmps = smoothed_velocity_mmps
                        
                    prev_valid_dist = min_dist
                    prev_valid_timestamp_ms = current_time_ms
                else:
                    prev_valid_dist = min_dist
                    prev_valid_timestamp_ms = current_time_ms
            else:
                prev_valid_dist = min_dist
                prev_valid_timestamp_ms = current_time_ms
                
        # TTC calculation
        ttc = 65535
        if velocity_mmps < 0:
            ttc_raw = (min_dist * 1000) // (-velocity_mmps)
            ttc = min(ttc_raw, 20000)
            
        # Teacher Labeling
        label = 0 # SAFE
        if min_dist < 250 or ttc < 1500:
            label = 2 # HAZARD
        elif min_dist < 350 or ttc < 3000 or (velocity_mmps < -200 and min_dist < 600):
            label = 1 # WARNING
            
        processed_rows.append([min_dist, velocity_mmps, ttc, label])
        
    return pd.DataFrame(processed_rows, columns=['MinDistance', 'ApproachVelocity', 'TTC', 'Label'])


file_list = ['Tof_test.csv'] + [f'Tof_test{i}.csv' for i in range(1, 10)]
accuracies = []

print("\n--- EVALUATING 10 RAW UNSEEN DATASETS ---")
for file in file_list:
    processed_df = process_file(file)
    # Save the labeled dataset for verification
    out_file = f"Processed_{file}"
    processed_df.to_csv(out_file, index=False)
    
    X = processed_df[['MinDistance', 'ApproachVelocity', 'TTC']].values
    y = processed_df['Label'].values
    
    X_scaled = X / max_vals
    
    preds = np.argmax(model.predict(X_scaled, batch_size=4096, verbose=0), axis=1)
    acc = accuracy_score(y, preds) * 100
    print(f"{file:<15} : {acc:>6.2f}%")
    accuracies.append(acc)

avg_acc = np.mean(accuracies)
print(f"\nAverage Accuracy across all 10 raw datasets: {avg_acc:.2f}%")
if avg_acc > 90.0:
    print("SUCCESS: Average accuracy is > 90%!")
else:
    print("WARNING: Average accuracy is below 90%.")
