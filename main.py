from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel
import numpy as np
import cv2
import tensorflow as tf

app = FastAPI()
templates = Jinja2Templates(directory="templates")

interpreter = tf.lite.Interpreter(
    model_path="models/movenet.tflite"
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


class PoseResponse(BaseModel):
    keypoints: list

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (192, 192))
    input_image = np.expand_dims(img, axis=0).astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()
    keypoints = interpreter.get_tensor(output_details[0]['index'])

    print("Keypoints shape:", keypoints.shape)

    return {"keypoints": keypoints.tolist()}