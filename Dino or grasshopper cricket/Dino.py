import serial
import pyautogui
import time

# Replace with your Arduino's serial port (COMx on Windows, /dev/ttyUSBx or /dev/ttyACMx on Linux/Mac)
SERIAL_PORT = 'COM6'  # Change this accordingly
BAUD_RATE = 115200

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Wait for Arduino to reset
    print("Connected to Arduino.")
except serial.SerialException:
    print("Could not connect to serial port.")
    exit()

while True:
    if ser.in_waiting:
        line = ser.readline().decode('utf-8').strip()
        print(f"Serial: {line}")  # optional: debug print
        if "DINO JUMP" in line:
            pyautogui.press('space')