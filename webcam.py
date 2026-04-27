from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from picamera2 import Picamera2
import cv2
import time

app = FastAPI()

# 1. Initialize PiCamera2
picam2 = Picamera2()

# 2. Configure the camera resolution 
# This replaces cv2.CAP_PROP_FRAME_WIDTH and HEIGHT
config = picam2.create_preview_configuration(main={"size": (1280, 720)})
picam2.configure(config)

# 3. Start the camera in the background
picam2.start()

def get_frame():
    try:
        # Capture a frame as a numpy array
        frame = picam2.capture_array()
        
        # PiCamera2 returns RGB by default, but OpenCV needs BGR to encode correctly.
        # Without this step, your red and blue colors will be swapped!
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame_bgr
    except Exception as e:
        print(f"Error capturing frame: {e}")
        return None

def generate_frames():
    while True:
        frame = get_frame()
        if frame is None:
            time.sleep(0.1)
            continue

        # Encode the BGR frame to JPEG
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
        <head>
            <title>Webcam Stream</title>
        </head>
        <body>
            <h1>FastAPI Webcam Stream (PiCamera2)</h1>
            <p><a href="/snapshot" target="_blank">Open snapshot</a></p>
            <img src="/video" width="800" />
        </body>
    </html>
    """


@app.get("/video")
def video():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/snapshot")
def snapshot():
    frame = get_frame()
    if frame is None:
        raise HTTPException(status_code=500, detail="Could not read frame from webcam")

    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode frame")

    return Response(content=buffer.tobytes(), media_type="image/jpeg")
