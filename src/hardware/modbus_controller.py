import time
import serial

# Command Frames
RELAY1_ON =  [1, 5, 0, 0, 0xFF, 0, 0x8C, 0x3A]  #  Kích hoạt mở relay 1
RELAY1_OFF = [1, 5, 0, 0, 0,    0, 0xCD, 0xCA]  #  Kích hoạt tắt relay 1

RELAY2_ON =  [1, 5, 0, 1, 0xFF, 0, 0xDD, 0xFA]  #  Kích hoạt mở relay 2
RELAY2_OFF = [1, 5, 0, 1, 0,    0, 0x9C, 0x0A]  #  Kích hoạt tắt relay 2

RELAY3_ON =  [1, 5, 0, 2, 0xFF, 0, 0x2D, 0xFA]  #  Kích hoạt mở relay 3
RELAY3_OFF = [1, 5, 0, 2, 0,    0, 0x6C, 0x0A]  #  Kích hoạt tắt relay 3

class ModbusController:
    def __init__(self, port):
        # Initialize serial connection
        self.ser = serial.Serial(port, baudrate=9600, timeout=1)
        print(f"[INFO] Connected to Modbus device on port {port}")

    def switch_relay_on(self, relay_number):
        if relay_number == 1:
            command = RELAY1_ON
        elif relay_number == 2:
            command = RELAY2_ON
        elif relay_number == 3:
            command = RELAY3_ON
        else:
            print(f"[ERROR] Invalid relay number: {relay_number}")
            return

        self._send_command(command)
        print(f"[INFO] Relay {relay_number} ON command sent.")

    def switch_relay_off(self, relay_number):
        if relay_number == 1:
            command = RELAY1_OFF
        elif relay_number == 2:
            command = RELAY2_OFF
        elif relay_number == 3:
            command = RELAY3_OFF
        else:
            print(f"[ERROR] Invalid relay number: {relay_number}")
            return

        self._send_command(command)
        print(f"[INFO] Relay {relay_number} OFF command sent.")
    
    def switch_all_on(self):
        self.switch_relay_on(1)
        time.sleep(0.1)  # Short delay to ensure commands are processed
        self.switch_relay_on(2)
        time.sleep(0.1)
        self.switch_relay_on(3)
        print("[INFO] All relays ON command sent.")

    def switch_all_off(self):
        self.switch_relay_off(1)
        time.sleep(0.1)  # Short delay to ensure commands are processed
        self.switch_relay_off(2)
        time.sleep(0.1)
        self.switch_relay_off(3)
        print("[INFO] All relays OFF command sent.")

    def close(self):
        self.ser.close()
        print("[INFO] Serial connection closed.")