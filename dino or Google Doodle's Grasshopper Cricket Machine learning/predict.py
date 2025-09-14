import serial
import numpy as np
from scipy import signal
import pandas as pd
import pickle
import pyautogui
import time
from collections import deque

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Sampling config
SAMPLING_RATE = 500
WINDOW_SIZE = 250         # 0.5s window
STEP_SIZE = 125       # 50% overlap

def setup_filters(sampling_rate):
    nyquist = 0.5 * sampling_rate
    low, high = 75 / nyquist, 150 / nyquist
    b_bandpass, a_bandpass = signal.butter(4, [low, high], btype='band')
    return b_bandpass, a_bandpass

def process_emg_data(data, b_bandpass, a_bandpass):
    return signal.filtfilt(b_bandpass, a_bandpass, data)

def calculate_segment_features(segment, sampling_rate):
    IEMG = np.sum(np.abs(segment))
    MAV = IEMG / len(segment)

    N = len(segment)
    MAV1 = np.sum([abs(n) * (1 if 0.25 * N <= i <= 0.75 * N else 0.5) for i, n in enumerate(segment)]) / N

    MAV2 = 0
    for i, n in enumerate(segment):
        if 0.25 * N <= i <= 0.75 * N:
            w = 1
        elif i < 0.25 * N:
            w = (4 * i) / N
        else:
            w = (4 * (i - N)) / N
        MAV2 += w * abs(n)
    MAV2 /= N

    SSI = np.sum(segment ** 2)
    VAR = np.var(segment, ddof=1)
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

def load_model_and_scaler():
    with open('model.pkl', 'rb') as f:
        clf = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return clf, scaler

def main():                                                                       
    ser = serial.Serial('COM6', 115200, timeout=1)
    clf, scaler = load_model_and_scaler()
    b_bandpass, a_bandpass = setup_filters(SAMPLING_RATE)
  
    buffer = deque(maxlen=WINDOW_SIZE + STEP_SIZE)  # 375 maxlen if you prefer 0.75s buffer, for now 375 (250 + 125)
    last_gesture = 0
    while True:
        try:
            raw_data = ser.readline().decode('utf-8').strip() #Read one line from the serial port ⇒ decode ⇒ strip whitespace
            if raw_data:
                try:
                    emg_value = float(raw_data)#Convert to float; if it fails, skip this sample.
                    buffer.append(emg_value)#Append to the rolling buffer.
                except ValueError:
                    continue

                if len(buffer) == WINDOW_SIZE + STEP_SIZE:
                    # Get the most recent 250 samples  out the collected 375 for processing
                    window = list(buffer)[-WINDOW_SIZE:]
                    window = np.array(window)

                    # Filter
                    filtered = process_emg_data(window, b_bandpass, a_bandpass)

                    # Feature extraction
                    features = calculate_segment_features(filtered, SAMPLING_RATE)
                    df = pd.DataFrame([features])

                    # Scale and predict
                    X_scaled = scaler.transform(df)
                    prediction = clf.predict(X_scaled)[0] #generally predict() function provides us with an array of probablities where the first element will have the highest probablity and hence we consider that as the final prediction
                    # print(f"prediction: {'rest' if prediction == 0 else 'movement'}")
                    # Trigger actions
                    now = time.time() 
                    if prediction == 1 and now - last_gesture > 0.5:
                        pyautogui.press('space')
                        print("space")
                        last_gesture = now      
                        #pyautogui.keyDown('w')
                        #time.sleep(1)                 # or pyautogui.click('space')
                        #Pyautogui.keyUp('w')

                    # Remove only STEP_SIZE samples for overlap
                    for i in range(STEP_SIZE):
                        buffer.popleft()

        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == '__main__':
    main()                                        