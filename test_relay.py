import serial
import time

# Adjust the port to match yours (e.g. /dev/ttyUSB0 or COM3)
ser = serial.Serial(port='/dev/ttyUSB0', baudrate=9600, bytesize=8, parity='N', stopbits=1)

# Example Modbus RTU frame to turn ON relay 1 (device address = 1)
# Function 05 (Write Single Coil): 0x01 0x05 0x00 0x00 0xFF 0x00 CRC_L CRC_H
frame_on  = bytes([0x01, 0x05, 0x00, 0x00, 0xFF, 0x00, 0x8C, 0x3A])
frame_off = bytes([0x01, 0x05, 0x00, 0x00, 0x00, 0x00, 0xCD, 0xCA])

for i in range (0,10):
    ser.write(frame_on)
    time.sleep(2)   
    ser.write(frame_off)
    time.sleep(2)

ser.close()
