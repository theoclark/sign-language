import keyboard
from time import time, sleep
import torch
from torch import nn
import numpy as np
import cv2
from torchvision import transforms

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=2),
            nn.Tanh(),
            nn.BatchNorm2d(6),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=2),
            nn.Tanh(),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=3, stride=2),
            nn.Tanh(),
            nn.BatchNorm2d(120),
            nn.Conv2d(in_channels=120, out_channels=120, kernel_size=3, padding=1, stride=2),
            nn.Tanh(),
            nn.BatchNorm2d(120),
            nn.Conv2d(in_channels=120, out_channels=120, kernel_size=3, padding=1, stride=2),
            nn.Tanh(),
            nn.BatchNorm2d(120),
            nn.Conv2d(in_channels=120, out_channels=120, kernel_size=3, padding=1, stride=1),
            nn.Tanh(),
            nn.BatchNorm2d(120),
            nn.Conv2d(in_channels=120, out_channels=120, kernel_size=3, padding=1, stride=1),
            nn.Tanh(),
            nn.BatchNorm2d(120),
            nn.Conv2d(in_channels=120, out_channels=120, kernel_size=3, padding=1, stride=1),
            nn.Tanh(),
            nn.BatchNorm2d(120),
            nn.Conv2d(in_channels=120, out_channels=120, kernel_size=3, padding=1, stride=1),
            nn.Tanh(),
            nn.BatchNorm2d(120),
            nn.Conv2d(in_channels=120, out_channels=120, kernel_size=3, padding=1, stride=1),
            nn.Tanh(),
            nn.BatchNorm2d(120),
            nn.Conv2d(in_channels=120, out_channels=64, kernel_size=3, stride=2),
            nn.Tanh(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=26, kernel_size=3, stride=2),
        )

    def forward(self, x):
        x = x.unsqueeze(dim=0)
        out = self.convnet(x)
        return out.squeeze()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
weights_path = './model_weights.pt'
frame_size = 420
image_size = 256
buffer_size = 30
threshold = 0.3
input_x, input_y = 20, 250
output_x, output_y = 800, 250
alphabet = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
buffer = torch.zeros((buffer_size, 26)).to(device)
model = Model()
model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
upsample = nn.Upsample((frame_size,frame_size))
downsample = transforms.Resize((image_size,image_size))

def score_frame(frame):
    with torch.no_grad():
        frame = torch.tensor(frame).permute((2,0,1)).type(torch.uint8)
        cropped_frame = transforms.functional.crop(frame,input_y,input_x,frame_size,frame_size),
        pred_frame = downsample(cropped_frame[0])
        image = pred_frame.float()/255
        logits = model(image.to(device))
        pred = nn.functional.softmax(logits,dim=-1)
        return pred, pred_frame

def generate_prediction(frame, buffer):
    pred, pred_frame = score_frame(frame)
    idx = pred.argmax()
    letter = alphabet[idx]
    buffer = torch.cat((buffer[1:],pred.unsqueeze(0)))
    mean_buffer = buffer.mean(dim=0)
    prob, idx = mean_buffer.max(0)
    if prob > threshold:
      return pred_frame, buffer, alphabet[idx]
    else:
      return pred_frame, buffer, "/"

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        pred_frame, buffer, pred = generate_prediction(frame, buffer)
        frame = torch.tensor(frame).unsqueeze(0).cpu().numpy()
        frame[:,output_y:output_y+frame_size,output_x:output_x+frame_size,:] = 0
        frame = frame.squeeze()
        white = (255,255,255)
        label_font = cv2.FONT_HERSHEY_SIMPLEX
        frame = cv2.putText(frame, pred, (output_x+100, output_y+frame_size-100), label_font, 8, white, 10)
        frame = cv2.rectangle(frame,(input_x, input_y), (input_x+frame_size, input_y+frame_size), white, 2)
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imshow('Input', frame)
        c = cv2.waitKey(1)
        if c == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
