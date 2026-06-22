import time

class HandController:
    """
    Interface for controlling the robotic hand.
    Currently prints commands to console.
    """
    def __init__(self, port=None, baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.current_state = "UNKNOWN"
        
        if self.port:
            print(f"Warning: Serial connection to {self.port} not fully implemented in this demo.")
            # self.serial = serial.Serial(port, baudrate)

    def send_command(self, command):
        """
        Send a command to the hand.
        command: "OPEN" or "CLOSE"
        """
        if command == self.current_state:
            return  # Avoid spamming the same command

        # Simulating sending command to robotic arm
        print(f"[CONTROL] -> Robotic Arm Actuator: {command}")
        self.current_state = command
        
        if self.port:
            # self.serial.write(command.encode())
            pass

if __name__ == "__main__":
    ctrl = HandController()
    ctrl.send_command("OPEN")
    time.sleep(1)
    ctrl.send_command("CLOSE")
