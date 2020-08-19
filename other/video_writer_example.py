import torch
import cv2
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter


summary_writer = SummaryWriter()


dir = 'videos'
os.makedirs(dir, exist_ok=True)

N = 1
T = 100
C = 3
H = 200
W = 500
video = np.random.randint(
    low=0,
    high=255,
    size=(N, T, C, H, W)
)
tensor = torch.from_numpy(video)
channel0 = 0
channel1 = 1
channel2 = 2
tensor[:, :, channel0] = torch.ones_like(tensor[:, :, channel0]) * 0
tensor[:, :, channel1] = torch.ones_like(tensor[:, :, channel1]) * 0
tensor[:, :, channel2] = torch.ones_like(tensor[:, :, channel2]) * 255

summary_writer.add_video(
    tag=dir,
    vid_tensor=tensor.type(torch.uint8),
    fps=10
)

summary_writer.close()