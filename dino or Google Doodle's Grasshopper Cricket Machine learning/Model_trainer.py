# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import pickle
# from sklearnex import patch_sklearn
# patch_sklearn()


# %% Load data
df = pd.read_csv('D:\\HCI-basics\\TASKS\\Task-5\\data.csv')  # Adjust file path as needed
df.drop(columns=df.columns[0], axis=1, inplace=True)
df.columns = ['raw_emg', 'label']

data = df['raw_emg']
labels_old = df['label']

sampling_rate = 500  # 500 Hz

# %% Bandpass filter (75–150 Hz)
lowcut, highcut = 75, 150
nyquist = 0.5 * sampling_rate
b_bandpass, a_bandpass = signal.butter(4, [lowcut / nyquist, highcut / nyquist], btype='band')


# Assume 500 samples per second (adjust if different)
samples_to_drop = 250

# Get all indices where label changes (from 0→1 or 1→0)
transition_indices = df[df['label'].diff().abs() == 1].index

# Create a set of indices to drop
drop_indices = set()
for idx in transition_indices:
    start = max(0, idx - samples_to_drop)
    end = min(len(df), idx + samples_to_drop + 1)
    drop_indices.update(range(start, end))

# Drop those indices
df_cleaned = df.drop(index=drop_indices).reset_index(drop=True)
df = df_cleaned
# Done!
print(f"Dropped {len(drop_indices)} rows")

# %% Feature extraction function
def calculate_segment_features(segment, sampling_rate):
    IEMG = np.sum(np.abs(segment))
    MAV = IEMG / len(segment)

    N = len(segment)
    MAV1 = 0
    for i, n in enumerate(segment):
        w = 1 if 0.25 * N <= i <= 0.75 * N else 0.5
        MAV1 += w * abs(n)
    MAV1 = MAV1 / N

    MAV2 = 0
    for i, n in enumerate(segment):
        if 0.25 * N <= i <= 0.75 * N:
            w = 1
        elif i < 0.25 * N:
            w = (4 * i) / N
        else:
            w = (4 * (i - N)) / N
        MAV2 += w * abs(n)
    MAV2 = MAV2 / N

    SSI = np.sum(segment ** 2)
    VAR = np.sum(segment ** 2) / (N - 1)
    RMS = np.sqrt(np.mean(segment ** 2))
    WL = np.sum(np.abs(np.diff(segment)))

    return {
        'IEMG': IEMG,
        'MAV': MAV,
        'MAV1': MAV1,
        'MAV2': MAV2,
        'SSI': SSI,
        'VAR': VAR,
        'RMS': RMS,
        'WL': WL
    }

# %% Segmenting with 0.5s windows (250 samples) and 50% overlap (125 samples)
features = []
labels = []

for i in range(0, len(data) - 250, 125):
    segment = data.loc[i:i+249]
    segment = pd.to_numeric(segment, errors='coerce')

    # Apply bandpass filter
    segment = signal.filtfilt(b_bandpass, a_bandpass, segment)

    # Extract features
    segment_features = calculate_segment_features(segment, sampling_rate)
    features.append(segment_features)
    labels.append(labels_old[i])

# %% Create DataFrame
columns = list(features[0].keys())
df_features = pd.DataFrame(features, columns=columns)
df_features['label'] = labels

# %% Save to CSV (optional)
df_features.to_csv('ready.csv', index=False)

# %% Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features.drop('label', axis=1))
df_scaled = pd.DataFrame(X_scaled, columns=columns)
df_scaled['label'] = df_features['label']

# %% Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df_scaled.drop('label', axis=1), df_scaled['label'], test_size=0.2, random_state=42
)

# %% Train model
model = RandomForestClassifier(n_estimators=10000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# %% Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Classification Accuracy: {accuracy:.4f}")

# %% Save model and scaler
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# %% Calibration
probabilities = model.predict_proba(X_test)[:, 1]
fop, mpv = calibration_curve(y_test, probabilities, n_bins=10)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(mpv, fop, marker='.')
plt.title("Calibration Curve")
plt.xlabel("Mean predicted value")
plt.ylabel("Fraction of positives")
plt.grid()
plt.show()

# %% Calibrate model
calibrator = CalibratedClassifierCV(model, cv=3)
calibrator.fit(X_train, y_train)
yhat = calibrator.predict(X_test)

# %% Calibrated accuracy
fop, mpv = calibration_curve(y_test, yhat, n_bins=10)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(mpv, fop, marker='.')
plt.title("Post-Calibration")
plt.grid()
plt.show()

accuracy = accuracy_score(y_test, yhat)
print(f"Post-Calibration Accuracy: {accuracy:.4f}")

# %% Full evaluation
y_true = df_scaled['label']
y_pred_all = model.predict(df_scaled.drop('label', axis=1))

print(f'Accuracy: {accuracy_score(y_true, y_pred_all):.4f}')
print(f'Precision: {precision_score(y_true, y_pred_all):.4f}')
print(f'Recall: {recall_score(y_true, y_pred_all):.4f}')
print(f'F1 Score: {f1_score(y_true, y_pred_all):.4f}')
print('Confusion Matrix:')
print(confusion_matrix(y_true, y_pred_all))
print('Classification Report:')
print(classification_report(y_true, y_pred_all))
