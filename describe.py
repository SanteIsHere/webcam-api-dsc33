import os
import threading
from io import BytesIO
from typing import Optional

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from google import genai
from PIL import Image
from dotenv import load_dotenv
from datetime import datetime  # To capture date and time
import requests  # To interact with OpenWeatherMap API

app = FastAPI(title="Webcam Snapshot Describer")
load_dotenv()

# -----------------------------
# Camera manager
# -----------------------------


from picamera2 import Picamera2

class Camera:
    def __init__(self, index: int = 0, width: int = 1280, height: int = 720):
        # Initialize the modern Picamera2 driver instead of cv2.VideoCapture
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(main={"size": (width, height)})
        self.picam2.configure(config)
        self.picam2.start()
        self.lock = threading.Lock()

    def read_frame(self):
        with self.lock:
            # Returns a NumPy array compatible with your existing OpenCV logic
            return self.picam2.capture_array()

    def get_jpeg_bytes(self) -> bytes:
        frame = self.read_frame()
        ok, buf = cv2.imencode(".jpg", frame)
        return buf.tobytes()

    def get_pil_image(self) -> Image.Image:
        frame = self.read_frame()
        # Picamera2 outputs RGB by default, so no cvtColor needed
        return Image.fromarray(frame)

    def release(self):
        with self.lock:
            self.picam2.stop()

camera: Optional[Camera] = None


# -----------------------------
# Gemini client
# -----------------------------
def get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    return genai.Client()


# -----------------------------
# FastAPI lifecycle
# -----------------------------


@app.on_event("startup")
def startup_event():
    global camera
    try:
        camera = Camera(index=0, width=1280, height=720)
    except Exception as e:
        # Keep app alive, but camera endpoints will fail until fixed
        print(f"Camera startup warning: {e}")


@app.on_event("shutdown")
def shutdown_event():
    global camera
    if camera is not None:
        camera.release()


# -----------------------------
# Endpoints
# -----------------------------


@app.get("/")
def root():
    return {
        "message": "Webcam snapshot describer is running",
        "endpoints": {
            "snapshot_jpg": "/snapshot.jpg",
            "describe": "/describe",
            "health": "/health",
        },
    }


@app.get("/health")
def health():
    return {"camera_ready": camera is not None}


@app.get("/snapshot.jpg")
def snapshot_jpg():
    if camera is None:
        raise HTTPException(status_code=500, detail="Camera not initialized")

    try:
        jpeg = camera.get_jpeg_bytes()
        return StreamingResponse(BytesIO(jpeg), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/describe")
def describe_snapshot():
    """
    Captures a snapshot from the webcam and asks Gemini to describe it.
    """
    if camera is None:
        raise HTTPException(status_code=500, detail="Camera not initialized")

    try:
        pil_img = camera.get_pil_image()
        client = get_gemini_client()

        # Get OWeather API key from env
        OW_API_KEY = os.environ.get("OW_API_KEY")

        # print(OW_API_KEY)

        # # Request weather data using OWeatherAPI
        city = "New Haven"
        payload = {"q": city, "APPID": OW_API_KEY}
        ow_req = requests.get(
            "https://api.openweathermap.org/data/2.5/weather", params=payload
        )

        # print(ow_req.json())
        ow_weather_data = ow_req.json()["weather"][0]
        ow_temp_data = ow_req.json()["main"]

        # print(ow_weather_data, ow_temp_data)

        # Function for K -> F temp conversion
        def k_to_f(k):
            return (k - 273.15) * 1.8 + 32

        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[
                "Describe this webcam snapshot clearly and concisely. "
                "Mention the main objects, people, setting, and anything notable. "
                "If text is visible, mention it only if readable.",
                pil_img,
            ],
        )

        return JSONResponse(
            {
                "description": response.text,
                "city": "New Haven",
                "datetime": datetime.now().strftime("%H:%M:%S, %Y-%m-%d"),
                "temperature": f"{k_to_f(ow_temp_data['temp']):.0f}F",
                "conditions": ow_weather_data["description"],
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
