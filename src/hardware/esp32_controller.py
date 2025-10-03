import serial, time
from .base_controller import HardwareController

class ESP32Controller(HardwareController):
    """
    PC-side controller to talk to ESP32 over USB/UART.
    Sends text-based commands like 'L1:ON', 'ALL:OFF',
    and waits for ACK/NACK responses.
    """
    def __init__(self, port, baudrate=115200, timeout:float=0.2):
       self.ser = serial.Serial(port, baudrate, timeout=timeout)
    
    def _send_command(self, cmd:str) -> bool:
        """Send a command string, expect ACK back"""
        if not self.ser.is_open:
            raise ConnectionError("Serial port not open")

        self.ser.write((cmd + '\n').encode('utf-8')) 
        print(f"[INFO] Command sent: {cmd}")
        self.ser.flush()
        
        time.sleep(0.1)
        resp = self.ser.readline().decode('utf-8').strip()

        if resp == "ACK":
            print(f"[INFO] Received ACK for command: {cmd}")
            return True
        elif resp == "NACK":
            print(f"[WARN] Received NACK for command: {cmd}")
            return False
        else:
            print(f"[WARN] No valid response for command: {cmd}, got: {resp}") 
            return False
    
    # High-level command 
    def switch(self, relay_number:int, state: bool):
        cmd = f"L{relay_number}:{'ON' if state else 'OFF'}"
        return self._send_command(cmd)
    
    def all_on(self):
        return self._send_command("ALL:ON")
    
    def all_off(self):
        return self._send_command("ALL:OFF")
    
    def close(self):    
        if self.ser.is_open:
            self.ser.close()
            print("[INFO] Serial port closed")
