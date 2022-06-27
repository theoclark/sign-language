import torch
from torch import nn
from torchvision import transforms
import cv2

# Adjustable Parameters
frame_size = 420                # Frame size for input and output boxes
image_size = 256                # Image size for the model
buffer_size = 30                # Number of previous predictions to keep in buffer
threshold = 0.3                 # Threshold above which to display prediction
input_x, input_y = 20, 250      # Top-left coordinate for the input box
output_x, output_y = 800, 250   # Top-left coordinate for the output box

class Model(nn.Module):
    """
    ConvNet for sign recognition.
    """
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
alphabet = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
buffer = torch.zeros((buffer_size, 26)).to(device)
model = Model()
model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
resize = transforms.Resize((image_size,image_size))

def score_frame(frame):
    """
    Crops the frame to the input box size and passes it to the model.
    Returns the predicted values for each letter fed through softmax.
    """
    with torch.no_grad():
        image = torch.tensor(frame).permute((2,0,1)).type(torch.uint8)
        image = transforms.functional.crop(image,input_y,input_x,frame_size,frame_size),
        image = resize(image[0])
        image = image.float()/255
        logits = model(image.to(device))
        pred = nn.functional.softmax(logits,dim=-1)
        return pred

def generate_prediction(frame, buffer):
    """
    Adds the latest prediction to the buffer (removing the oldest value).
    Buffer is then averaged and the maximum value returned.
    If the predicted value is above the threshold the prediction is displayed.
    """
    pred = score_frame(frame)
    buffer = torch.cat((buffer[1:],pred.unsqueeze(0)))
    mean_buffer = buffer.mean(dim=0)
    prob, idx = mean_buffer.max(0)
    if prob > threshold:
      return buffer, alphabet[idx]
    else:
      return buffer, "/"

# Open input video stream
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

with torch.no_grad():
    while True:
        # Get current frame
        ret, frame = cap.read()
        # Generate prediction
        buffer, pred = generate_prediction(frame, buffer)
        # Edit and show output frame
        frame[output_y:output_y+frame_size,output_x:output_x+frame_size,:] = 0
        rgb = (255,255,255)
        label_font = cv2.FONT_HERSHEY_SIMPLEX
        frame = cv2.putText(frame, pred, (output_x+100, output_y+frame_size-100), label_font, 8, rgb, 10)
        frame = cv2.rectangle(frame,(input_x, input_y), (input_x+frame_size, input_y+frame_size), rgb, 2)
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imshow('Sign Language Detector', frame)
        c = cv2.waitKey(1)
