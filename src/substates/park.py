class ParkSubstate:
    """Park substate handling hold and spotlight tracking."""

    DEFAULT = "Hold"

    def __init__(self):
        self.state = self.DEFAULT

    def enter(self):
        self.state = self.DEFAULT

    def exit(self):
        pass

    def handle_event(self, event: str):
        if event in ("obstacle_detected", "max_distance"):
            self.state = "Spotlight"
        return self.state

    def current(self):
        return self.state
