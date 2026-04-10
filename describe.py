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


class Camera:
    def __init__(
        self, index: int = 0, width: Optional[int] = None, height: Optional[int] = None
    ):
        self.index = index
        self.cap = cv2.VideoCapture(index)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam at index {index}")

        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.lock = threading.Lock()

    def read_frame(self):
        with self.lock:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                raise RuntimeError("Failed to read frame from webcam")
            return frame

    def get_jpeg_bytes(self) -> bytes:
        frame = self.read_frame()
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("Failed to encode frame as JPEG")
        return buf.tobytes()

    def get_pil_image(self) -> Image.Image:
        frame = self.read_frame()
        # OpenCV is BGR; PIL expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def release(self):
        with self.lock:
            if self.cap is not None:
                self.cap.release()


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
