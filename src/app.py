import argparse
import asyncio
import json
import logging
import os
import time
import requests
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

    def __init__(self, source="http://192.168.68.58:8080/video", model_path=None, width=640, height=480, fps=20):
        super().__init__()
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps

        self.session = None
        self.stream = None
        self.stream_iter = None
        self.bytes = b""

        self._open_stream()

        # Initialize gesture detector and state machine
        self.gesture_detector = GestureDetector()
        self.state_machine = RobotStateMachine(validation_delay=1.5)

        if USE_LEGACY_MEDIAPIPE:
            self.mp_hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.use_tasks = False
        else:
            model_path = model_path or os.getenv(MODEL_ENV_NAME) or DEFAULT_TASK_MODEL
            if not model_path or not os.path.isfile(model_path):
                raise FileNotFoundError(
                    'MediaPipe 0.10+ requires a hand landmarker task bundle. '
                    'Download or provide hand_landmarker.task and pass --model <path> '
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

    def _close_stream(self):
        if self.stream is not None:
            try:
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        if self.session is not None:
            try:
                self.session.close()
            except Exception:
                pass
            self.session = None

    def _open_stream(self):
        self._close_stream()
        self.session = requests.Session()
        self.stream = self.session.get(
            self.source,
            stream=True,
            timeout=(5, 15),
            headers={
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache',
                'User-Agent': 'OpponentRobot/1.0'
            }
        )
        if self.stream.status_code != 200:
            raise RuntimeError("Cannot open IP camera stream")
        self.stream_iter = self.stream.iter_content(chunk_size=4096)

    def _reopen_stream(self):
        logging.warning("Reopening MJPEG stream: %s", self.source)
        try:
            self._open_stream()
        except Exception as exc:
            logging.warning("Failed to reopen MJPEG stream: %s", exc)
            time.sleep(1)
            self._open_stream()

    def read(self):
        while True:
            try:
                for chunk in self.stream_iter:
                    if not chunk:
                        continue
                    self.bytes += chunk
                    a = self.bytes.find(b'\xff\xd8')
                    if a == -1:
                        continue
                    b = self.bytes.find(b'\xff\xd9', a + 2)
                    if b == -1:
                        continue
                    jpg = self.bytes[a:b+2]
                    self.bytes = self.bytes[b+2:]
                    img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        img = cv2.resize(img, (self.width, self.height))
                        return True, img
                logging.warning("MJPEG stream ended unexpectedly, reconnecting...")
            except Exception as exc:
                logging.warning("MJPEG read error: %s", exc)
            self._reopen_stream()

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        loop = asyncio.get_event_loop()
        ret, frame = await loop.run_in_executor(None, self.read)

        if not ret:
            # return a black frame if capture failed
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # rotate 90 degrees left (counter-clockwise)
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

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

        # Detect gesture
        gesture = None
        if landmarks:
            gesture = self.gesture_detector.process(landmarks, handedness_labels)

        # Remember previous state so we can react to mode/substate transitions
        prev_state = self.state_machine.get_state()

        # Update state machine with detected gesture (main-mode detection has precedence)
        state = self.state_machine.update(gesture)

        # If we just entered Command main mode, reset follow center so the next
        # open-palm detection sets the center at the current palm position.
        try:
            prev_main = prev_state.get('main_mode')
            new_main = state.get('main_mode')
            prev_sub = prev_state.get('substate')
            new_sub = state.get('substate')
        except Exception:
            prev_main = prev_sub = new_main = new_sub = None

        if prev_main != new_main and new_main == 'Command':
            # reset so detector will initialize center on first palm
            self.gesture_detector.reset_follow_center()

        # If we were in a directional follow substate and just returned to the
        # Follow substate (e.g. halted), reset center so it will be set on next palm
        directional = ('Left', 'Right', 'Back', 'Forward')
        if prev_sub in directional and new_sub == 'Follow':
            self.gesture_detector.reset_follow_center()
        # If we left Follow mode, clear the center so an open palm in another
        # mode is recognized as the main 'open_palm' gesture (not a substate).
        if prev_main == 'Command' and new_main != 'Command':
            self.gesture_detector.reset_follow_center()

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
            if state['main_mode'] in ('Park', 'Command') and state.get('substate'):
                mode_text += f" ({state['substate']})"
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
    root_dir = os.path.dirname(__file__)
    app.router.add_get('/', lambda r: web.FileResponse(os.path.join(root_dir, 'static', 'index.html')))

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
                'substate': state['substate'],
                'validation_progress': state['validation_progress']
            }))
        return web.Response(content_type='application/json', text=json.dumps({
            'gesture': None,
            'mode': None,
            'substate': None,
            'validation_progress': 0
        }))

    app.router.add_get('/gesture', get_gesture)
    app.on_shutdown.append(on_shutdown)
    return app


if __name__ == '__main__':
    # Parse CLI arguments and start the aiohttp server.
    parser = argparse.ArgumentParser(description='Run WebRTC OpenCV/MediaPipe server')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', default=8081, type=int)
    parser.add_argument(
        '--source',
        default="0",
        type=str,
        help='Camera index (e.g. 0) or HTTP URL (e.g. http://192.168.x.x:8080/video)'
    )
    parser.add_argument('--model', default=None,
                        help='Path to hand_landmarker.task for MediaPipe 0.10+ installs')
    args = parser.parse_args()

    # Convert numeric strings to int for webcam indices
    source_arg = args.source
    if source_arg.isdigit():
        source_arg = int(source_arg)

    app = create_app(source=source_arg, model_path=args.model)
    web.run_app(app, host=args.host, port=args.port)
