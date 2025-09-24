from abc import ABC, abstractmethod

class HardwareController(ABC):
    """
    Abstract base class for hardware controllers.
    Defines the common interface for controlling relays.
    """
    @abstractmethod
    def switch(self, relay_number: int, state: bool):
        """
        Switches a specific relay to the given state (ON/OFF).
        :param relay_number: The number of the relay to control.
        :param state: True for ON, False for OFF.
        """
        pass

    @abstractmethod
    def all_on(self):
        # Switches all connected relays ON.
        pass

    @abstractmethod
    def all_off(self):
        # Switches all connected relays OFF.
        pass

    @abstractmethod
    def close(self):
        # Closes the connection to the hardware.
        pass
