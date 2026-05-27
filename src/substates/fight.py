class FightSubstate:
    """Fight substate handling tracking and evasion transitions."""

    DEFAULT = "Tracking"

    def __init__(self):
        self.state = self.DEFAULT

    def enter(self):
        self.state = self.DEFAULT

    def exit(self):
        pass

    def handle_event(self, event: str):
        if event == "obstacle_detected":
            self.state = "Evasion"
        elif event == "obstacle_clear":
            self.state = "Tracking"
        return self.state

    def current(self):
        return self.state
