from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import StreamingResponse

import cv2
import torch
import numpy as np
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

# Setup paths for model files
classes_to_filter = None

opt  = {
    "weights": "runs/train/exp/weights/best.pt",
    "yaml"   : "Trash-5/data.yaml",
    "img-size": 640,
    "conf-thres": 0.25,
    "iou-thres" : 0.45,
    "device" : 'cpu',
    "classes" : classes_to_filter
}

# Load model
device = select_device(opt['device'])
half = device.type != 'cpu'
model = attempt_load(opt['weights'], map_location=device)
imgsz = check_img_size(opt['img-size'], s=model.stride.max())
if half:
    model.half()

if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

webcam = cv2.VideoCapture(0)
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/about_us", tags=["pages"])
async def about_us(request: Request):
    return templates.TemplateResponse("about_us.html", {"request": request})

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def generate():
    while True:
        ret_val, img = webcam.read()
        if not ret_val:
            print("Failed to grab frame from webcam. Is it connected?")
            continue
        img0 = img.copy()
        img = letterbox(img, new_shape=opt['img-size'])[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = time_synchronized()
        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, opt['conf-thres'], opt['iou-thres'], classes=opt['classes'], agnostic=False)
        t2 = time_synchronized()

        for i, det in enumerate(pred):
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)

        (flag, encodedImage) = cv2.imencode(".jpg", img0)
        if not flag:
            continue

        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace;boundary=frame")