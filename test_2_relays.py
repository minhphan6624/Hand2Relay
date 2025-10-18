import serial, time

ser = serial.Serial('/dev/ttyUSB0', 9600, 8, 'N', 1)

# Relay 1 ON/OFF
ch1_on  = bytes([0x01,0x05,0x00,0x00,0xFF,0x00,0x8C,0x3A])
ch1_off = bytes([0x01,0x05,0x00,0x00,0x00,0x00,0xCD,0xCA])

# Relay 2 ON/OFF
ch2_on  = bytes([0x01,0x05,0x00,0x01,0xFF,0x00,0xDD,0xFA])
ch2_off = bytes([0x01,0x05,0x00,0x01,0x00,0x00,0x9C,0x0A])

# Test pattern
# ser.write(ch1_on); time.sleep(1)
# ser.write(ch1_off); time.sleep(1)
# ser.write(ch2_on); time.sleep(1)
# ser.write(ch2_off); time.sleep(1)

for i in range(5):
    ser.write(ch1_on); time.sleep(0.05); ser.write(ch2_on)
    time.sleep(1)
    ser.write(ch1_off); time.sleep(0.05); ser.write(ch2_off)
    time.sleep(1)


ser.close()
