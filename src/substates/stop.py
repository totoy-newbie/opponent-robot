class IdleSubstate:
    """Idle substate representing a halted motor state."""

    DEFAULT = "Halt"

    def __init__(self):
        self.state = self.DEFAULT

    def enter(self):
        self.state = self.DEFAULT

    def exit(self):
        pass

    def current(self):
        return self.state
