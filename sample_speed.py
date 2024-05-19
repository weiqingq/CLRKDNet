import argparse
import torch
import cv2
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from clrkd.utils.config import Config
from clrkd.engine.runner import Runner
import time

# Setup argument parser
parser = argparse.ArgumentParser(description="Run inference speed test with specified configuration and model weights.")
parser.add_argument('--config', type=str, required=True, help='Path to the model configuration file.')
parser.add_argument('--load_from', type=str, required=True, help='Path to the model weights file.')

# Parse arguments
args = parser.parse_args()

# Configuration setup using arguments
cfg = Config.fromfile(args.config)
cfg.gpus = 1

cfg.load_from = args.load_from
cfg.resume_from = ''
cfg.finetune_from = ''
cfg.seed = 1234
cfg.distillation = False
img_w = 800
img_h = 320

image = torch.ones((1, 3, img_h, img_w)).cuda()

cudnn.benchmark = True

runner = Runner(cfg)

# Warm-up step
for _ in range(500):
    _ = runner.inference(image)

start_time = time.time()

# Record time for each inference to calculate max FPS
inference_times = []

# Main inference loop
for i in tqdm(range(3000)):
    start_time = time.time()
    output = runner.inference(image)
    inference_time = time.time() - start_time
    inference_times.append(inference_time)

# Calculate total time taken for inferences
total_time = sum(inference_times)

# Calculate frames per second (FPS) for average and max
average_fps = 3000 / total_time
max_fps = 1 / min(inference_times)  # max FPS is the inverse of the minimum time taken for a single inference

print(f"Average Inference FPS: {average_fps}")
print(f"Maximum Inference FPS: {max_fps}")
print(f"Total Time: {total_time}")