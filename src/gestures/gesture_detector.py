"""Gesture detector using MediaPipe hand landmarks to classify hand poses and movements.

This module provides `GestureDetector` with a lightweight API:
- `process(landmarks)` -> returns a detected gesture string or `None`.

Detected gestures:
- 'open_palm' (open palm facing camera at or near the saved center position)
- 'palm_down' (open palm facing downward)
- 'left', 'right', 'forward', 'back' (open palm moved away from follow center)
- 'fist' (fist)
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
        """Initialize the gesture detector."""
        self.history = deque(maxlen=max_history)
        self.motion_thresh = motion_thresh
        self.follow_center = None
        self.follow_threshold = 0.08

    def _count_fingers(self, hand_landmarks_list: List[Any], hand_idx: int, is_right: bool) -> int:
        """Count extended fingers in a hand."""
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
        """Detect if hand is pointing left or right (only index finger extended)."""
        finger_count = self._count_fingers(hand_landmarks_list, hand_idx, is_right)
        if finger_count != 1:
            return None
        
        hand_lm = hand_landmarks_list[hand_idx]
        hand_center_x = hand_lm[9].x
        index_tip_x = hand_lm[8].x
        dx = index_tip_x - hand_center_x
        
        if abs(dx) < 0.05:
            return None
        return 'park_left' if dx < 0 else 'park_right'

    def _detect_follow_direction(self, centroid: tuple) -> str:
        """Detect follow substate directions relative to the saved palm center."""
        if self.follow_center is None:
            self.follow_center = centroid
            return 'open_palm'

        dx = centroid[0] - self.follow_center[0]
        dy = centroid[1] - self.follow_center[1]

        if abs(dx) < self.follow_threshold and abs(dy) < self.follow_threshold:
            return 'open_palm'

        if abs(dx) > abs(dy):
            return 'right' if dx > 0 else 'left'
        return 'forward' if dy > 0 else 'back'

    def _detect_palm_orientation(self, hand_landmarks_list: List[Any], hand_idx: int) -> Optional[str]:
        """Distinguish open palm (facing camera) vs palm down (facing ground)."""
        hand_lm = hand_landmarks_list[hand_idx]
        wrist_z = hand_lm[0].z
        fingertip_ids = [8, 12, 16, 20]
        avg_tip_z = sum(hand_lm[i].z for i in fingertip_ids if i < len(hand_lm)) / len(fingertip_ids)

        # Open palm: fingertips closer to camera than wrist
        if avg_tip_z < wrist_z - 0.02:
            return 'open_palm'
        # Palm down: fingertips farther than wrist
        elif avg_tip_z > wrist_z + 0.02:
            return 'palm_down'
        return None

    def reset_follow_center(self):
        """Reset the follow center so the next open palm reinitializes it."""
        self.follow_center = None

    def process(self, hand_landmarks_list: List[Any], handedness_list: List[str]) -> Optional[str]:
        """Process hand landmarks and return a detected gesture."""
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

        avg_cx = sum(h['centroid'][0] for h in hands) / len(hands)
        avg_cy = sum(h['centroid'][1] for h in hands) / len(hands)
        now = time.time()
        self.history.append({'time': now, 'cx': avg_cx, 'cy': avg_cy, 'hands': hands})

        # Two hands: two fists -> fight mode
        if len(hands) >= 2:
            if all(h['fingers'] == 0 for h in hands):
                return 'fight'

        # Single hand gestures
        elif len(hands) >= 1:
            h0 = hands[0]
            
            # Open palm (4+ fingers) -> check orientation
            if h0['fingers'] >= 4:
                orientation = self._detect_palm_orientation(hand_landmarks_list, 0)
                if orientation == 'open_palm':
                    return self._detect_follow_direction(h0['centroid'])
                elif orientation == 'palm_down':
                    self.reset_follow_center()
                    return 'palm_down'
            
            # Closed fist (0 fingers)
            elif h0['fingers'] == 0:
                self.reset_follow_center()
                return 'fist'
            
            # Pointing (1 finger)
            else:
                pointing = self._detect_pointing(hand_landmarks_list, 0, h0['is_right'])
                if pointing:
                    return pointing

        return None
