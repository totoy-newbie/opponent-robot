import argparse
import asyncio
import json
import logging
import os
import time
from aiohttp import web

import cv2
import numpy as np
import av

from src.gestures.gesture_detector import GestureDetector
from src.state_machine import RobotStateMachine

# Global runtime configuration for MediaPipe compatibility.
# The app supports both legacy `mediapipe.solutions` and the newer
# MediaPipe Tasks API from mediapipe>=0.10.
USE_LEGACY_MEDIAPIPE = False
MODEL_ENV_NAME = 'MP_HAND_LANDMARKER_TASK'
DEFAULT_TASK_MODEL = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'hand_landmarker.task')
)

# Detect which MediaPipe API is available at runtime.
# Prefer the legacy solutions API when available for backwards compatibility.
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    USE_LEGACY_MEDIAPIPE = True
except (AttributeError, ModuleNotFoundError):
    # Fall back to the newer MediaPipe Tasks API for mediapipe>=0.10.
    try:
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
        from mediapipe.tasks.python.vision.core import image as mp_image
        from mediapipe.tasks.python.core import base_options as mp_base_options
        from mediapipe.tasks.python.vision.core import vision_task_running_mode as mp_running_mode
    except Exception as exc:
        raise ImportError(
            'MediaPipe is installed but neither legacy solutions nor task APIs are available. '
            'Use a compatible mediapipe version or install a task model bundle. '
            'If you are on Python 3.12+, install mediapipe>=0.10 and pass --model <path-to-hand_landmarker.task>.'
        ) from exc

from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack

logging.basicConfig(level=logging.INFO)

# Hand landmark connections for drawing the hand skeleton.
# Indices follow MediaPipe hand landmark ordering:
# 0 = wrist
# 1-4 = thumb (CMC to tip)
# 5-8 = index finger
# 9-12 = middle finger
# 13-16 = ring finger
# 17-20 = pinky finger
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (5, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (9, 13), (13, 14), (14, 15), (15, 16),
    # Pinky finger
    (13, 17), (17, 18), (18, 19), (19, 20),
    # Wrist to pinky base
    (0, 17)
]


class OpenCVMediaTrack(VideoStreamTrack):
    """A VideoStreamTrack that captures frames from OpenCV, runs MediaPipe hands,
    annotates the frames and yields them for WebRTC streaming.

    This class wraps a local camera source and makes it available as a WebRTC
    video track for the browser client.
    """

    def __init__(self, source=0, model_path=None, width=640, height=480, fps=20):
        super().__init__()
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.fps = fps

        # Initialize gesture detector and state machine
        self.gesture_detector = GestureDetector()
        self.state_machine = RobotStateMachine(validation_delay=2.0)

        if USE_LEGACY_MEDIAPIPE:
            # Legacy MediaPipe solutions API uses the Hands class directly.
            self.mp_hands = mp_hands.Hands(static_image_mode=False,
                                           max_num_hands=2,
                                           min_detection_confidence=0.5,
                                           min_tracking_confidence=0.5)
            self.use_tasks = False
        else:
            # For MediaPipe Tasks API, load the hand_landmarker.task model bundle.
            model_path = model_path or os.getenv(MODEL_ENV_NAME) or DEFAULT_TASK_MODEL
            if not model_path or not os.path.isfile(model_path):
                raise FileNotFoundError(
                    'MediaPipe 0.10+ requires a hand landmarker task bundle. '
                    'Download or provide hand_landmarker.task and pass --model <path> ' \
                    f'or set {MODEL_ENV_NAME}.'
                )

            base_options = mp_base_options.BaseOptions(model_asset_path=model_path)
            options = HandLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_running_mode.VisionTaskRunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.mp_hands = HandLandmarker.create_from_options(options)
            self.use_tasks = True

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        loop = asyncio.get_event_loop()
        ret, frame = await loop.run_in_executor(None, self.cap.read)
        if not ret:
            # return a black frame if capture failed
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # mirror the camera feed so it behaves like a front-facing webcam
        frame = cv2.flip(frame, 1)

        # process with MediaPipe (expects RGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.use_tasks:
            mp_image_obj = mp_image.Image(mp_image.ImageFormat.SRGB, rgb)
            results = self.mp_hands.detect_for_video(
                mp_image_obj,
                int(time.time() * 1000),
            )
            landmarks = results.hand_landmarks if results else None
            # Extract handedness labels from Tasks API results
            handedness_labels = []
            if results and hasattr(results, 'handedness') and results.handedness:
                for hand_class in results.handedness:
                    if hand_class:
                        handedness_labels.append(hand_class[0].category_name)
        else:
            results = self.mp_hands.process(rgb)
            landmarks = results.multi_hand_landmarks if results else None
            # Extract handedness labels from legacy API
            handedness_labels = []
            if results and results.multi_handedness:
                for hnd in results.multi_handedness:
                    handedness_labels.append(hnd.classification[0].label)

        # Detect gesture and update state machine
        gesture = None
        if landmarks:
            gesture = self.gesture_detector.process(landmarks, handedness_labels)

        # Update state machine with detected gesture
        state = self.state_machine.update(gesture)

        # draw landmarks and skeleton overlay
        if landmarks:
            for hand_landmarks in landmarks:
                landmark_list = getattr(hand_landmarks, 'landmark', hand_landmarks)
                points = []
                for lm in landmark_list:
                    x = int(lm.x * frame.shape[1])
                    y = int(lm.y * frame.shape[0])
                    points.append((x, y))
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

                for start, end in HAND_CONNECTIONS:
                    if start < len(points) and end < len(points):
                        cv2.line(frame, points[start], points[end], (0, 255, 0), 2)

        # Display detected gesture and mode on the video
        display_y = 30
        if state['gesture']:
            gesture_text = f"Gesture: {state['gesture']} ({state['validation_progress']}%)"
            cv2.putText(frame, gesture_text, (10, display_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            display_y += 30

        if state['main_mode']:
            mode_text = f"Mode: {state['main_mode']}"
            cv2.putText(frame, mode_text, (10, display_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3)

        # convert to av.VideoFrame
        video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame


# Keep track of active peer connections so they can be closed cleanly.
pcs = set()

# Track the current video track for gesture access
current_video_track = None


async def on_shutdown(app):
    # Close all active peer connections when the web app shuts down.
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)


def create_app(source=0, model_path=None):
    # Create the aiohttp application and define the WebRTC signaling endpoint.
    app = web.Application()
    app.router.add_get('/', lambda r: web.FileResponse('./src/static/index.html'))

    async def offer(request):
        # Handle browser WebRTC offers. This endpoint receives SDP from the client,
        # creates a peer connection, attaches the local OpenCV video track, and
        # returns the server answer.
        params = await request.json()
        offer = RTCSessionDescription(sdp=params['sdp'], type=params['type'])

        pc = RTCPeerConnection()
        pcs.add(pc)

        @pc.on('connectionstatechange')
        def on_connectionstatechange():
            logging.info('Connection state: %s', pc.connectionState)
            if pc.connectionState == 'failed':
                asyncio.ensure_future(pc.close())

        video = OpenCVMediaTrack(source=source, model_path=model_path)
        pc.addTrack(video)

        # Store reference to current video track for gesture access
        global current_video_track
        current_video_track = video

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.Response(content_type='application/json', text=json.dumps({
            'sdp': pc.localDescription.sdp,
            'type': pc.localDescription.type
        }))

    app.router.add_post('/offer', offer)

    async def get_gesture(request):
        # Return current gesture and mode state as JSON for browser display
        global current_video_track
        if current_video_track:
            state = current_video_track.state_machine.get_state()
            return web.Response(content_type='application/json', text=json.dumps({
                'gesture': state['gesture'],
                'mode': state['main_mode'],
                'validation_progress': state['validation_progress']
            }))
        return web.Response(content_type='application/json', text=json.dumps({
            'gesture': None,
            'mode': None,
            'validation_progress': 0
        }))

    app.router.add_get('/gesture', get_gesture)
    app.on_shutdown.append(on_shutdown)
    return app


if __name__ == '__main__':
    # Parse CLI arguments and start the aiohttp server.
    # Use --source to select the camera index and --model to provide a MediaPipe task bundle.
    parser = argparse.ArgumentParser(description='Run WebRTC OpenCV/MediaPipe server')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', default=8080, type=int)
    parser.add_argument('--source', default=0, type=int)
    parser.add_argument('--model', default=None,
                        help='Path to hand_landmarker.task for MediaPipe 0.10+ installs')
    args = parser.parse_args()

    app = create_app(source=args.source, model_path=args.model)
    web.run_app(app, host=args.host, port=args.port)
