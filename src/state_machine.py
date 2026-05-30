"""Hierarchical state machine for managing gesture-to-mode transitions with validation delay.

This module implements the opponent robot's state machine with:
- Main modes: Command, Park, Fight, Stop
- Sub-states for each mode (e.g., Command → Command/Forward/Left/Right/Back)
- Gesture-triggered transitions as defined in doc/design/*.puml

State hierarchy:
├─ Command (Follow, Forward, Left, Right, Back)
├─ Park (ParkLeft, ParkRight, Spotlight)
├─ Fight (Tracking, Evasion)
└─ Stop (Halt)
"""
import time
from typing import Optional
from enum import Enum


class MainMode(Enum):
    """Main operational modes."""
    COMMAND = "Command"
    PARK = "Park"
    FIGHT = "Fight"
    STOP = "Stop"


class CommandSubstate(Enum):
    """Follow mode sub-states."""
    FOLLOW = "Follow"
    FORWARD = "Forward"
    LEFT = "Left"
    RIGHT = "Right"
    BACK = "Back"


class ParkSubstate(Enum):
    """Park mode sub-states."""
    PARK_LEFT = "ParkLeft"
    PARK_RIGHT = "ParkRight"
    SPOTLIGHT = "Spotlight"


class FightSubstate(Enum):
    """Fight mode sub-states."""
    TRACKING = "Tracking"
    EVASION = "Evasion"


class StopSubstate(Enum):
    """Stop mode sub-states."""
    HALT = "Halt"


class RobotStateMachine:
    """Hierarchical state machine following the opponent robot design.
    
    Manages main modes and their sub-states, with gesture-triggered transitions.
    """

    # Gesture to main mode mapping
    GESTURE_TO_MAIN_MODE = {
        'follow': MainMode.COMMAND,
        'stop': MainMode.STOP,
        'fight': MainMode.FIGHT,
        'park_left': MainMode.PARK,
        'park_right': MainMode.PARK,
    }

    PARK_SUBSTATE_GESTURES = {
        'park_left': ParkSubstate.PARK_LEFT,
        'park_right': ParkSubstate.PARK_RIGHT,
    }

    COMMAND_SUBSTATE_GESTURES = {
        'follow': CommandSubstate.FOLLOW,
        'left': CommandSubstate.LEFT,
        'right': CommandSubstate.RIGHT,
        'back': CommandSubstate.BACK,
        'forward': CommandSubstate.FORWARD,
        'swipe_left': CommandSubstate.LEFT,
        'swipe_right': CommandSubstate.RIGHT,
        'swipe_up': CommandSubstate.BACK,
        'swipe_down': CommandSubstate.FORWARD,
    }

    def __init__(self, validation_delay: float = 1.5):
        """Initialize the state machine.

        Args:
            validation_delay: Time in seconds to hold a gesture before confirming mode.
        """
        self.validation_delay = validation_delay
        # Gestures that should immediately switch main mode without waiting
        # for the validation_delay when detected (useful for responsive control)
        self.immediate_main_gestures = {'follow'}
        
        # Gesture tracking
        self.current_gesture: Optional[str] = None
        self.gesture_start_time: Optional[float] = None
        self.pending_mode: Optional[MainMode] = None
        
        # Main mode state
        self.main_mode: MainMode = MainMode.COMMAND
        
        # Sub-state tracking for each mode
        self.command_substate: CommandSubstate = CommandSubstate.FOLLOW
        self.park_substate: ParkSubstate = ParkSubstate.PARK_LEFT
        self.fight_substate: FightSubstate = FightSubstate.TRACKING
        self.stop_substate: StopSubstate = StopSubstate.HALT

    def update(self, detected_gesture: Optional[str]) -> dict:
        """Update state machine with a newly detected gesture.

        Args:
            detected_gesture: Gesture string from detector, or None if no gesture.

        Returns:
            Dict with state information:
            - 'main_mode': Current main mode (Command/Park/Fight/Stop)
            - 'substate': Current sub-state of the main mode
            - 'gesture': Current detected gesture
            - 'validation_progress': Percentage of validation time elapsed (0-100)
        """
        now = time.time()

        # Main-mode detection takes precedence. If the detected gesture maps
        # to a main mode, only handle main-mode validation/transition here and
        # do not interpret it as a substate gesture.
        main_candidate = self.GESTURE_TO_MAIN_MODE.get(detected_gesture)
        # Only run main-mode validation if the detected gesture maps to a
        # different main mode than the one we're currently in. If it maps
        # to the same main mode, treat it as a substate gesture below.
        if main_candidate is not None and main_candidate != self.main_mode:
            # Immediate transition for specific gestures (no validation delay)
            if detected_gesture in self.immediate_main_gestures:
                self._transition_to_mode(main_candidate, detected_gesture)
                # reset any ongoing gesture tracking
                self.current_gesture = None
                self.gesture_start_time = None
                self.pending_mode = None
                return self._get_state()

            # Otherwise handle main-mode gesture validation and transition
            if detected_gesture != self.current_gesture:
                self.current_gesture = detected_gesture
                self.gesture_start_time = now if detected_gesture else None
                self.pending_mode = None
            elif detected_gesture and self.gesture_start_time:
                elapsed = now - self.gesture_start_time
                if elapsed >= self.validation_delay:
                    new_mode = main_candidate
                    self._transition_to_mode(new_mode, detected_gesture)
                    self.pending_mode = None

            return self._get_state()

        # Only interpret substate gestures when already inside the corresponding
        # main mode. Substate gestures should not trigger main-mode logic.
        if self.main_mode == MainMode.COMMAND and detected_gesture in self.COMMAND_SUBSTATE_GESTURES:
            self.command_substate = self.COMMAND_SUBSTATE_GESTURES[detected_gesture]
        if self.main_mode == MainMode.PARK and detected_gesture in self.PARK_SUBSTATE_GESTURES:
            self.park_substate = self.PARK_SUBSTATE_GESTURES[detected_gesture]
        # Note: 'stop' maps to a main mode and was handled above; do not treat
        # it as a follow-substate gesture here.

        return self._get_state()

    def _transition_to_mode(self, new_mode: MainMode, gesture: str):
        """Transition to a new main mode based on gesture."""
        self.main_mode = new_mode

        if new_mode == MainMode.COMMAND:
            self.command_substate = CommandSubstate.FOLLOW
        elif new_mode == MainMode.PARK:
            # Default park substate on entering Park; specific left/right
            # gestures are handled as park-substate gestures while in Park.
            self.park_substate = ParkSubstate.PARK_LEFT
        elif new_mode == MainMode.FIGHT:
            self.fight_substate = FightSubstate.TRACKING
        elif new_mode == MainMode.STOP:
            self.stop_substate = StopSubstate.HALT

    def _get_state(self) -> dict:
        """Get current full state."""
        # Determine current substate based on main mode
        if self.main_mode == MainMode.COMMAND:
            current_substate = self.command_substate.value
        elif self.main_mode == MainMode.PARK:
            current_substate = self.park_substate.value
        elif self.main_mode == MainMode.FIGHT:
            current_substate = self.fight_substate.value
        elif self.main_mode == MainMode.STOP:
            current_substate = self.stop_substate.value
        else:
            current_substate = "Unknown"

        # Calculate validation progress percentage
        validation_progress = 0
        if self.current_gesture and self.gesture_start_time:
            elapsed = time.time() - self.gesture_start_time
            validation_progress = min(100, int((elapsed / self.validation_delay) * 100))

        return {
            'main_mode': self.main_mode.value,
            'substate': current_substate,
            'gesture': self.current_gesture,
            'validation_progress': validation_progress
        }

    def get_state(self) -> dict:
        """Get current state without updating."""
        return self._get_state()

    def reset(self):
        """Reset the state machine to initial state."""
        self.current_gesture = None
        self.gesture_start_time = None
        self.pending_mode = None
        self.main_mode = MainMode.COMMAND
        self.command_substate = CommandSubstate.FOLLOW
        self.park_substate = ParkSubstate.PARK_LEFT
        self.fight_substate = FightSubstate.TRACKING
        self.stop_substate = StopSubstate.HALT
