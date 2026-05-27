class CommandSubstate:
    """Follow substate handling directional follow/swap gestures."""

    DEFAULT = "Follow"

    def __init__(self):
        self.state = self.DEFAULT

    def enter(self):
        self.state = self.DEFAULT

    def exit(self):
        pass

    def handle_gesture(self, gesture: str):
        mapping = {
            "swipe_left": "Left",
            "swipe_right": "Right",
            "swipe_up": "Back",
            "swipe_down": "Forward",
            "close_fist": self.DEFAULT,
        }
        new = mapping.get(gesture)
        if new:
            self.state = new
        return self.state

    def current(self):
        return self.state
