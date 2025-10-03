from typing import Dict
from hardware.base_controller import HardwareController

class GestureActionMapper:
    """
    Maps recognized gestures to specific hardware control actions.
    """
    def __init__(self, controller: HardwareController, gestures_config: Dict[int, str]):
        self.controller = controller
        self.gestures_config = {name: idx for idx, name in gestures_config.items()} # Invert for easy lookup
        self.state = {1: False, 2: False, 3: False} # Track relay states for Modbus

    def _toggle_relay(self, relay_number: int, state: bool):
        """Internal helper to toggle a single relay."""
        self.controller.switch(relay_number, state)
        self.state[relay_number] = state
        print(f"[ACTION] Relay {relay_number} {'ON' if state else 'OFF'}")

    def execute_action(self, gesture_name: str, simulation: bool = False):
        """
        Executes the hardware action corresponding to the recognized gesture.
        :param gesture_name: The name of the recognized gesture.
        :param simulation: If True, prints the action instead of executing on hardware.
        """
        if simulation:
            print(f"[SIMULATION] Executing action for: {gesture_name}")
            return

        if gesture_name == "light1_on":
            self._toggle_relay(1, True)
        elif gesture_name == "light1_off":
            self._toggle_relay(1, False)
        elif gesture_name == "light2_on":
            self._toggle_relay(2, True)
        elif gesture_name == "light2_off":
            self._toggle_relay(2, False)
        elif gesture_name == "all_on" or gesture_name == "turn_on":
            self.controller.all_on()
            for r in self.state: self.state[r] = True
            print("[ACTION] All relays ON")
        elif gesture_name == "all_off" or gesture_name == "turn_off":
            self.controller.all_off()
            for r in self.state: self.state[r] = False
            print("[ACTION] All relays OFF")
        else:
            print(f"[WARN] No action defined for gesture: {gesture_name}")
