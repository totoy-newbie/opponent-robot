import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    intro=mo.md("""
    <h1 id="intro">1. Introduction</h1>
    <p><strong>Purpose:</strong><br>
    Design and implement a mobile sparring partner robot (boxing dummy on mecanum wheel platform) that responds to user gestures, tracks movement, and ensures safety through local reflexes and lightweight logging.</p>

    <p><strong>Scope:</strong><br>
    The system integrates vision (Mini PC), control (ESP32), sensors (ultrasonic, lidar), and communication (Bluetooth/Wi‑Fi) to simulate fight training scenarios with real‑time responsiveness.</p>
    """)

    return (intro,)


@app.cell
def _(mo):
    overview=mo.md("""
    <h1 id="overview">2. System Overview</h1>
    <ul>
      <li><strong>Opponent Robot:</strong> Century BOB dummy mounted on mecanum‑wheel base.</li>
      <li><strong>Mini PC (Linux):</strong> Vision, gesture recognition, mode logic, communication, logging, video streaming.</li>
      <li><strong>ESP32 Controller:</strong> Motor commands, sensor polling, obstacle avoidance, logging.</li>
      <li><strong>Remote Laptop:</strong> Displays live video stream via WebRTC.</li>
    </ul>
    """)

    return (overview,)


@app.cell
def _(mo):
    functional=mo.md("""
    <h1 id="functional">3. Functional Requirements</h1>

    <h3>Modes of Operation</h3>
    <ul>
      <li>Follow Mode: Open palm → Position adjustment.</li>
      <li>Park Mode: Two fingers → Lateral movement.</li>
      <li>Fight Mode: Two fists → Dynamic tracking & evasion.</li>
      <li>Stop Mode: Palm down → Immediate halt.</li>
    </ul>

    <h3>Gesture Recognition</h3>
    <p>Mediapipe Hand Tracking, latency <150 ms, default to safe mode if unrecognized.</p>

    <h3>Motion Control</h3>
    <p>Mecanum wheels, JSON commands over Bluetooth/Wi‑Fi, smooth ramping, local obstacle reflex overrides.</p>

    <h3>Communication</h3>
    <p>JSON packets with checksum + sequence ID, heartbeat every 2s, fallback to Wi‑Fi if Bluetooth drops.</p>

    <h3>Video Streaming</h3>
    <p>WebRTC (<300 ms latency), MJPEG fallback.</p>

    <h3>Logging</h3>
    <p>Lightweight JSON logs: sensor readings, motor commands, events.</p>
    """)

    return (functional,)


@app.cell
def _(mo):
    nonfunctional=mo.md("""
    <h1 id="nonfunctional">4. Non-Functional Requirements</h1>
    <ul>
      <li>Safety: Obstacle detection mandatory, emergency stop overrides all.</li>
      <li>Performance: Gesture latency <150 ms, video latency <300 ms.</li>
      <li>Reliability: Auto-reconnect communication, local reflexes independent of Mini PC.</li>
      <li>Extensibility: Future actuators, analytics dashboard.</li>
      <li>Maintainability: Modular code, JSON schema validation.</li>
    </ul>
    """)

    return (nonfunctional,)


@app.cell
def _(mo):
    architecture=mo.md("""
    <h1 id="architecture">5. Architecture & Testing</h1>

    <h3>Architecture</h3>
    <ul>
      <li>Mini PC: vision.py, control.py, comms.py, logger.py.</li>
      <li>ESP32: MotorDriver.h, Ultrasonic.h, Lidar.h, Safety.h, Logger.h.</li>
      <li>Communication: Command, log, heartbeat packets.</li>
      <li>Simulation: Python arena with visualization, real‑time animation, gesture input.</li>
    </ul>

    <h3>Testing</h3>
    <ul>
      <li>Unit Tests: Gesture recognition, sensor polling.</li>
      <li>Integration Tests: Command latency, obstacle reflex, Bluetooth drop → Wi‑Fi fallback.</li>
      <li>Simulation Tests: Fight‑mode tracking, auto‑switch stop/resume.</li>
      <li>Field Tests: Real sparring sessions with safety validation.</li>
    </ul>
    """)

    return (architecture,)


@app.cell
def _(mo):
    future=mo.md("""
    <h1 id="future">6. Future Enhancements</h1>
    <ul>
      <li>Actuated dummy arms for attack simulation.</li>
      <li>Battery monitoring and power management.</li>
      <li>Analytics dashboard for training insights.</li>
      <li>AI‑driven predictive evasive maneuvers.</li>
    </ul>
    """)

    return (future,)


@app.cell
def _(architecture, functional, future, intro, mo, nonfunctional, overview):
    mo.md(f"""
    <style>
      body {{
        display: flex;
        font-family: Arial, sans-serif;
      }}
      .sidebar {{
        width: 220px;
        background: #f4f4f4;
        padding: 15px;
        height: 100vh;
        position: fixed;
        overflow-y: auto;
      }}
      .sidebar a {{
        display: block;
        padding: 8px;
        margin-bottom: 5px;
        text-decoration: none;
        color: #333;
        border-radius: 4px;
      }}
      .sidebar a:hover {{
        background: #ddd;
      }}
      .content {{
        margin-left: 240px;
        padding: 20px;
        flex-grow: 1;
      }}
    </style>

    <div class="sidebar">
      <h3>Navigation</h3>
      <a href="#intro">1. Introduction</a>
      <a href="#overview">2. System Overview</a>
      <a href="#functional">3. Functional Requirements</a>
      <a href="#nonfunctional">4. Non-Functional Requirements</a>
      <a href="#architecture">5. Architecture & Testing</a>
      <a href="#future">6. Future Enhancements</a>
    </div>

    <div class="content">
      {intro}
      {overview}
      {functional}
      {nonfunctional}
      {architecture}
      {future}
    </div>
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
