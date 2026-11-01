# 📡 1D Time-of-Flight (ToF) Multi-Sensor Array Dataset

A comprehensive empirical dataset captured from a **4-channel 1D Time-of-Flight (ToF) optical sensor array**, curated for real-time robotic obstacle avoidance, distance estimation, approach velocity tracking, Time-to-Collision (TTC) forecasting, and zone safety classification.

Designed and tested in conjunction with hardware-accelerated FPGA implementations (Xilinx Vivado) and predictive maintenance neural networks.

---

## 📌 Overview & Sensor Configuration

The dataset captures multi-point distance measurements from four spatially distributed 1D ToF laser-ranging sensors:

- **Channels:** `TOF1_mm`, `TOF2_mm`, `TOF3_mm`, `TOF4_mm` (millimeter resolution).
- **Out-of-Range Sentinel:** `8190` indicates out-of-range, target loss, or sensor occlusion.
- **Sampling Frequency:** ~30 Hz real-time acquisition.
- **Engineered Kinematic Features:**
  - $d_{\min}$: Minimum distance across all operational sensors.
  - $v_{\text{approach}}$: Exponentially smoothed relative velocity ($\text{mm/s}$).
  - $\text{TTC}$: Real-time Time-to-Collision ($\text{s}$).
  - Safety Zone Labels (`Safe`, `Warning`, `Danger`).

---

## 📂 Dataset Taxonomy & Directory Structure

To eliminate data redundancy, the repository is partitioned into structured directories according to test scenario and processing stage:

```text
1D-ToF-Dataset/
├── benchmarks_10_scenarios/        # 10 Edge-case & functional benchmark scenarios
│   ├── Dataset_01_SlowApproach.csv
│   ├── Dataset_02_FastApproach.csv
│   ├── Dataset_03_MovingAway.csv
│   ├── Dataset_04_Stationary.csv
│   ├── Dataset_05_OneSensorFailure.csv
│   ├── Dataset_06_TwoSensorFailure.csv
│   ├── Dataset_07_RandomNoise.csv
│   ├── Dataset_08_TemporaryBlockage.csv
│   ├── Dataset_09_AllSensorsFailure.csv
│   └── Dataset_10_Recovery.csv
├── calibration_fixed_distances/    # Ground-truth static distance calibration
│   ├── 300.csv (300 mm static baseline)
│   ├── 400.csv (400 mm static baseline)
│   ├── 500.csv (500 mm static baseline)
│   ├── 600.csv (600 mm static baseline)
│   ├── 700.csv (700 mm static baseline)
│   └── 800.csv (800 mm static baseline)
├── raw_dynamic_sessions/           # 10 Raw dynamic physical test sessions
│   ├── Tof_test.csv  ... Tof_test9.csv
│   ├── Tof_actual.csv
│   ├── Tof_data.csv
│   └── warning.csv
├── processed_features/             # Filtered & feature-engineered datasets
│   └── Processed_Tof_test.csv ... Processed_Tof_test9.csv
├── merged_training/                # Aggregated datasets for machine learning & FPGA models
│   ├── Tof_merged_11_sessions_17field.csv (63k rows with 17 engineered features)
│   ├── merged_session_and_zone_spec.csv   (58k rows)
│   ├── zone_spec_dataset.csv              (24k rows)
│   ├── safe_far_approach.csv              (20k rows)
│   ├── warning_receding_300_400.csv       (12k rows)
│   └── tof_session.csv                    (34k rows)
├── scripts/                        # Evaluation, preprocessing, and confusion matrix scripts
│   ├── process_and_evaluate_raw.py
│   ├── get_cm_acc.py
│   └── requirements.txt
├── LICENSE                         # MIT License
└── README.md                       # Documentation
```

---

## 📊 Data Schema

### 1. Raw Data Schema (`raw_dynamic_sessions/`, `calibration_fixed_distances/`, `benchmarks_10_scenarios/`)

| Column Name | Type | Units | Description |
| :--- | :--- | :--- | :--- |
| `Timestamp_s` | Float | Seconds | High-resolution elapsed session time |
| `TOF1_mm` | Integer | mm | Channel 1 distance measurement (8190 = invalid) |
| `TOF2_mm` | Integer | mm | Channel 2 distance measurement (8190 = invalid) |
| `TOF3_mm` | Integer | mm | Channel 3 distance measurement (8190 = invalid) |
| `TOF4_mm` | Integer | mm | Channel 4 distance measurement (8190 = invalid) |

### 2. Processed Feature Schema (`processed_features/`, `merged_training/`)

| Column Name | Type | Units | Description |
| :--- | :--- | :--- | :--- |
| `MinDistance` | Float | mm | $\min(\{ d_i \mid d_i < 8190 \})$ |
| `ApproachVelocity` | Float | mm/s | Exponentially smoothed derivative of distance |
| `TTC` | Float | Seconds | Time-To-Collision: $\frac{\text{MinDistance}}{\lvert \text{ApproachVelocity} \rvert}$ |
| `Label` | Integer / Categorical | — | Classified hazard zone index (Danger / Warning / Safe) |

---

## 🧮 Mathematical Formulation

For a multi-sensor sample vector $\mathbf{d}(t) = [d_1(t), d_2(t), d_3(t), d_4(t)]^T$:

1. **Active Measurement Filtering:**
   $$\mathcal{D}(t) = \{ d_i(t) \mid d_i(t) < 8190 \}$$
   $$d_{\min}(t) = \begin{cases} \min(\mathcal{D}(t)), & \mathcal{D}(t) \neq \emptyset \\ 8190, & \mathcal{D}(t) = \emptyset \end{cases}$$

2. **Velocity Estimation:**
   $$v(t) = \alpha \cdot \frac{d_{\min}(t) - d_{\min}(t - \Delta t)}{\Delta t} + (1 - \alpha) \cdot v(t - \Delta t)$$

3. **Time-To-Collision (TTC):**
   $$\text{TTC}(t) = \begin{cases} \frac{d_{\min}(t)}{-v(t)}, & v(t) < 0 \text{ (approaching)} \\ \infty, & v(t) \ge 0 \text{ (receding or stationary)} \end{cases}$$

---

## 🛠️ Usage & Evaluation

```bash
cd scripts
pip install -r requirements.txt
python process_and_evaluate_raw.py
python get_cm_acc.py
```

---

## 📄 License

This dataset and associated code are licensed under the [MIT License](LICENSE).
