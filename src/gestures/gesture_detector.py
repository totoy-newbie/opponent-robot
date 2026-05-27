"""Gesture detector using MediaPipe hand landmarks to classify hand poses and movements.

This module provides `GestureDetector` with a lightweight API:
- `process(landmarks)` -> returns a detected gesture string or `None`.

Detected gestures:
- 'command' (open palm, all fingers extended)
- 'stop' (palm down, fist)
- 'fight' (two fists)
- 'park_left' (point left with index finger)
- 'park_right' (point right with index finger)

The heuristics are intentionally simple and intended as a starting point
for integration with the state machine. Improve thresholds and logic for
production use.
"""
import time
from collections import deque
from typing import Optional, List, Any


class GestureDetector:
    def __init__(self, max_history=5, motion_thresh=0.05):
        """Initialize the gesture detector.
        
        Args:
            max_history: Number of frames to track for motion-based gestures
            motion_thresh: Minimum centroid movement to consider a swipe
        """
        self.history = deque(maxlen=max_history)
        self.motion_thresh = motion_thresh

    def _count_fingers(self, hand_landmarks_list: List[Any], hand_idx: int, is_right: bool) -> int:
        """Count extended fingers in a hand.
        
        Indices of finger tips: 4 (thumb), 8 (index), 12 (middle), 16 (ring), 20 (pinky).
        Each tip is compared to the PIP (proximal interphalangeal) joint below it.
        """
        tips_ids = [4, 8, 12, 16, 20]
        count = 0
        hand_lm = hand_landmarks_list[hand_idx]
        
        for tip_id in tips_ids:
            if tip_id >= len(hand_lm):
                continue
            tip = hand_lm[tip_id]
            pip = hand_lm[tip_id - 2]
            
            # Thumb uses x comparison; others use y
            if tip_id == 4:
                if is_right and tip.x < pip.x:
                    count += 1
                elif not is_right and tip.x > pip.x:
                    count += 1
            else:
                if tip.y < pip.y:
                    count += 1
        return count

    def _centroid(self, hand_landmarks_list: List[Any], hand_idx: int) -> tuple:
        """Calculate the centroid of a hand."""
        hand_lm = hand_landmarks_list[hand_idx]
        xs = [lm.x for lm in hand_lm]
        ys = [lm.y for lm in hand_lm]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def _detect_pointing(self, hand_landmarks_list: List[Any], hand_idx: int, is_right: bool) -> Optional[str]:
        """Detect if hand is pointing left or right (only index finger extended).
        
        Uses the vector from hand center (middle finger MCP) to index finger tip
        to determine pointing direction. The hand must have only the index finger
        extended for this to register as a valid pointing gesture.
        """
        finger_count = self._count_fingers(hand_landmarks_list, hand_idx, is_right)
        
        # Check if only index finger is extended (count == 1)
        if finger_count != 1:
            return None
        
        hand_lm = hand_landmarks_list[hand_idx]
        
        # Use middle finger MCP (landmark 9) as hand center for better reference
        hand_center_x = hand_lm[9].x
        hand_center_y = hand_lm[9].y
        
        # Index finger tip (landmark 8) position
        index_tip_x = hand_lm[8].x
        index_tip_y = hand_lm[8].y
        
        # Calculate the vector from hand center to index finger tip
        dx = index_tip_x - hand_center_x
        dy = index_tip_y - hand_center_y
        
        # Require significant horizontal component (> 0.05 normalized units)
        # to avoid false positives from minor position variations
        if abs(dx) < 0.05:
            return None
        
        # Pointing left if index tip is to the left of hand center, right otherwise
        if dx < 0:
            return 'park_left'
        else:
            return 'park_right'

    def process(self, hand_landmarks_list: List[Any], handedness_list: List[str]) -> Optional[str]:
        """Process hand landmarks and return a detected gesture.

        Args:
            hand_landmarks_list: List of hand landmark objects from MediaPipe
            handedness_list: List of handedness strings ('Left' or 'Right')

        Returns:
            Gesture string or None if no clear gesture detected.
        """
        detected = None
        
        if not hand_landmarks_list:
            self.history.append(None)
            return None

        hands = []
        for i, hand_lm in enumerate(hand_landmarks_list):
            is_right = handedness_list[i] == 'Right' if i < len(handedness_list) else True
            fingers = self._count_fingers(hand_landmarks_list, i, is_right)
            centroid = self._centroid(hand_landmarks_list, i)
            hands.append({
                'fingers': fingers,
                'centroid': centroid,
                'is_right': is_right,
                'handedness': handedness_list[i] if i < len(handedness_list) else 'Right'
            })

        # Store centroid history for motion detection
        centers = [h['centroid'] for h in hands]
        avg_cx = sum(c[0] for c in centers) / len(centers)
        avg_cy = sum(c[1] for c in centers) / len(centers)
        now = time.time()
        self.history.append({'time': now, 'cx': avg_cx, 'cy': avg_cy, 'hands': hands})

        # Gesture heuristics
        # Two hands: two fists -> fight mode
        if len(hands) >= 2:
            fcounts = [h['fingers'] for h in hands]
            if all(fc == 0 for fc in fcounts):
                return 'fight'

        # Single hand gestures
        if len(hands) >= 1:
            h0 = hands[0]
            
            # Open palm (4+ fingers) -> command mode
            if h0['fingers'] >= 4:
                return 'command'
            
            # Closed fist (0 fingers) -> stop mode
            if h0['fingers'] == 0:
                return 'stop'
            
            # Pointing (1 finger) -> park mode (left or right)
            pointing = self._detect_pointing(hand_landmarks_list, 0, h0['is_right'])
            if pointing:
                return pointing

        return None
