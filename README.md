# Opponent Robot — WebRTC Vision Boilerplate

This scaffold provides a minimal Python server that captures video with OpenCV, runs MediaPipe hand tracking, and streams annotated frames to a browser via WebRTC.

Quick start

1. Create and activate a virtual environment (optional but recommended).

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Run the server (serves the client at http://0.0.0.0:8080/):

```bash
python -m src.app
```

Notes
- The server uses your default webcam as the capture source. Pass `--source` to `src.app` to change.
- `aiortc` and `av` may require system libraries (FFmpeg). If installation fails, follow platform-specific instructions for `av`/FFmpeg.
# opponent-robot
A body opponent dummy with mecanum wheels and a camera for tracking humans.
