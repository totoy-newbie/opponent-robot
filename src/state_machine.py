"""Hierarchical state machine for managing gesture-to-mode transitions with validation delay.

This module implements the opponent robot's state machine with:
- Main modes: Follow, Park, Fight, Stop
- Sub-states for each mode (e.g., Follow → Follow/Forward/Left/Right/Back)
- Gesture-triggered transitions as defined in doc/design/*.puml

State hierarchy:
├─ Follow (Follow, Forward, Left, Right, Back)
├─ Park (ParkLeft, ParkRight, Spotlight)
├─ Fight (Tracking, Evasion)
└─ Stop (Halt)
"""
import time
from typing import Optional
from enum import Enum


class MainMode(Enum):
    """Main operational modes."""
    FOLLOW = "Follow"
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
        'follow': MainMode.FOLLOW,
        'stop': MainMode.STOP,
        'fight': MainMode.FIGHT,
        'park_left': MainMode.PARK,
        'park_right': MainMode.PARK,
    }

    def __init__(self, validation_delay: float = 2.0):
        """Initialize the state machine.

        Args:
            validation_delay: Time in seconds to hold a gesture before confirming mode.
        """
        self.validation_delay = validation_delay
        
        # Gesture tracking
        self.current_gesture: Optional[str] = None
        self.gesture_start_time: Optional[float] = None
        self.pending_mode: Optional[MainMode] = None
        
        # Main mode state
        self.main_mode: MainMode = MainMode.FOLLOW
        
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
            - 'main_mode': Current main mode (Follow/Park/Fight/Stop)
            - 'substate': Current sub-state of the main mode
            - 'gesture': Current detected gesture
            - 'validation_progress': Percentage of validation time elapsed (0-100)
        """
        now = time.time()

        # Handle gesture change or timeout
        if detected_gesture != self.current_gesture:
            # Gesture changed or ended; reset validation
            self.current_gesture = detected_gesture
            self.gesture_start_time = now if detected_gesture else None
            self.pending_mode = None
        elif detected_gesture and self.gesture_start_time:
            # Same gesture held; check if validation window expired
            elapsed = now - self.gesture_start_time
            if elapsed >= self.validation_delay:
                # Gesture held long enough; transition to new mode
                new_mode = self.GESTURE_TO_MAIN_MODE.get(detected_gesture)
                if new_mode:
                    self._transition_to_mode(new_mode, detected_gesture)
                    self.pending_mode = None

        return self._get_state()

    def _transition_to_mode(self, new_mode: MainMode, gesture: str):
        """Transition to a new main mode based on gesture."""
        self.main_mode = new_mode

        if new_mode == MainMode.FOLLOW:
            self.command_substate = CommandSubstate.FOLLOW
        elif new_mode == MainMode.PARK:
            # Initialize park substate based on pointing direction
            self.park_substate = (
                ParkSubstate.PARK_LEFT if gesture == 'park_left'
                else ParkSubstate.PARK_RIGHT
            )
        elif new_mode == MainMode.FIGHT:
            self.fight_substate = FightSubstate.TRACKING
        elif new_mode == MainMode.STOP:
            self.stop_substate = StopSubstate.HALT

    def _get_state(self) -> dict:
        """Get current full state."""
        # Determine current substate based on main mode
        if self.main_mode == MainMode.FOLLOW:
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
        self.main_mode = MainMode.FOLLOW
        self.command_substate = CommandSubstate.FOLLOW
        self.park_substate = ParkSubstate.PARK_LEFT
        self.fight_substate = FightSubstate.TRACKING
        self.stop_substate = StopSubstate.HALT
