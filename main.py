from fileinput import filename
import uvicorn
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv
import os

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from models import SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr

model_file = "backend/models/SRCNN_x4.pth"
scale = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SRCNN().to(device)
model.load_state_dict(torch.load(model_file, map_location=device))
model.eval()


load_dotenv()

port = os.getenv("PORT", "8000")

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "Hello World"}

@app.post("/upload")
def upload_file(file: UploadFile):

    # with open("request", "wb") as f:
    #     # f.write(str(file))
    #     # f.write(str(file.filename))
    #     f.write(file.file.read())

    filename = file.filename
    image = pil_image.open(file.file).convert("RGB")
    
    image = np.array(image).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(image)

    y = ycbcr[..., 0]
    y /= 255.0
    y = torch.from_numpy(y).to(device)
    y = y.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        preds = model(y).clamp(0.0, 1.0)

    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    output.save(filename.replace(".", "_srcnn_x{}.".format(scale)))
    return FileResponse(filename.replace(".", "_srcnn_x{}.".format(scale)))


if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=int(port), reload=True)