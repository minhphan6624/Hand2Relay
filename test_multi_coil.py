import serial, struct, time

# --- Function to compute CRC16 (Modbus RTU) ---
def crc16(data: bytes):
    crc = 0xFFFF
    for pos in data:
        crc ^= pos
        for _ in range(8):
            if crc & 1:
                crc >>= 1
                crc ^= 0xA001
            else:
                crc >>= 1
    return struct.pack('<H', crc)  # little-endian

# --- Open RS-485 serial port ---
ser = serial.Serial('/dev/ttyUSB0', 9600, 8, 'N', 1, timeout=1)

# --- Build "turn ON both relays" frame (coils 0 & 1 ON) ---
frame_on = bytes([0x01, 0x0F, 0x00, 0x00, 0x00, 0x02, 0x01, 0x03])
frame_on += crc16(frame_on)

# --- Build "turn OFF both relays" frame (coils 0 & 1 OFF) ---
frame_off = bytes([0x01, 0x0F, 0x00, 0x00, 0x00, 0x02, 0x01, 0x00])
frame_off += crc16(frame_off)

# --- Send ON command ---
print("Sending ON frame:", frame_on.hex(' '))
ser.write(frame_on)
time.sleep(2)  # Keep ON for 2 seconds

# --- Send OFF command ---
print("Sending OFF frame:", frame_off.hex(' '))
ser.write(frame_off)

ser.close()
print("Done.")
