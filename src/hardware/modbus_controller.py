import time
import serial
from hardware.base_controller import HardwareController

# Command Frames
RELAY1_ON =  [1, 5, 0, 0, 0xFF, 0, 0x8C, 0x3A]  
RELAY1_OFF = [1, 5, 0, 0, 0,    0, 0xCD, 0xCA]  

RELAY2_ON =  [1, 5, 0, 1, 0xFF, 0, 0xDD, 0xFA]  
RELAY2_OFF = [1, 5, 0, 1, 0,    0, 0x9C, 0x0A]  

RELAY3_ON =  [1, 5, 0, 2, 0xFF, 0, 0x2D, 0xFA]  
RELAY3_OFF = [1, 5, 0, 2, 0,    0, 0x6C, 0x0A]  

CMD_DICT = {
    (1, True): RELAY1_ON,
    (1, False): RELAY1_OFF,
    (2, True): RELAY2_ON,
    (2, False): RELAY2_OFF,
    (3, True): RELAY3_ON,
    (3, False): RELAY3_OFF
}

class ModbusController(HardwareController):
    def __init__(self, port):
        # Initialize serial connection
        self.ser = serial.Serial(port, baudrate=9600, timeout=1)
        print(f"[INFO] Connected to Modbus device on port {port}")

    def _send_command(self, command):
        try:
            self.ser.write(bytearray(command))
            print(f"[INFO] Command sent: {command}")
        except Exception as e:
            print(f"[ERROR] Failed to send command: {e}")

    def switch(self, relay_number:int, state: bool):
        if (relay_number, state) in CMD_DICT:
            command = CMD_DICT[(relay_number, state)]
            self._send_command(command)
            print(f"[INFO] Relay {relay_number} {'ON' if state else 'OFF'} command sent.")
        else:
            print(f"[ERROR] Invalid command for Relay {relay_number} {'ON' if state else 'OFF'}.")

    def all_on(self):
        for relay in [1, 2, 3]:
            self.switch(relay, True)
        print(f"[INFO] All relays ON command sent.")

    def all_off(self):
        for relay in [1, 2, 3]:
            self.switch(relay, False)
        print(f"[INFO] All relays OFF command sent.")
    
    def close(self):
        if self.ser.is_open:
            self.ser.close()
            print("[INFO] Serial connection closed.")
