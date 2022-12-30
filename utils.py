import torch
from torchvision import transforms

import numpy as np
from skimage.color import rgb2lab, lab2rgb
import skimage.transform
from PIL import Image

import os
from tqdm import tqdm
from moviepy.editor import VideoFileClip
import cv2
from pathlib import Path

from pdb import set_trace

def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)

SIZE = 256

def get_L(img):
    img = transforms.Resize((SIZE, SIZE),  transforms.InterpolationMode.BICUBIC)(img)
    img = np.array(img)
    img_lab = rgb2lab(img).astype("float32")
    img_lab = transforms.ToTensor()(img_lab)
    L = img_lab[[0], ...] / 50. - 1. # Between -1 and 1

    return L

def get_predictions(model, L):
    # model.L = L.to(model.device)
    model.L = L.to(torch.device('cpu'))
    model.forward()
    fake_color = model.fake_color.detach()
    fake_imgs = lab_to_rgb(L, fake_color)

    return fake_imgs

def inference_model_img(model, img):
    L = get_L(img)
    L = L[None] # put in list
    fake_imgs = get_predictions(model, L)
    fake_img = fake_imgs[0] # get out of list
    resized_fake_img = skimage.transform.resize(fake_img, img.size[::-1])

    return resized_fake_img

# def inference_model_batch_imgs(model, imgs):
#     L = get_L(imgs)
#     fake_imgs = get_predictions(model, L)
#     resized_fake_imgs = [skimage.transform.resize(fake_img, imgs[0].size[::-1]) for fake_img in fake_imgs]
    
#     return resized_fake_imgs

def inference_model_vid(model, path_input: str, fps=None):

    input_video = VideoFileClip(path_input)
    # nframes = input_video.reader.nframes
    # print(f"There are {nframes} frames to process")
    
    if fps:
        used_fps = fps
        nframes = np.round(fps * input_video.duration)
    else:
        used_fps = input_video.fps
        nframes = input_video.reader.nframes
    print(f"Colorizing output of FPS: {fps}, Nframes: {nframes}.")
    
    frames = input_video.iter_frames(fps=used_fps)
    
    colorized_frames = []
    for frame in tqdm(frames, total=nframes):
        color_frame = inference_model_img(model, Image.fromarray(frame))
        colorized_frames.append(color_frame)

    height, width = colorized_frames[0].shape[:2]
    size = (width, height)

    # copy audio from input
    input_audio = input_video.audio

    # create tmp path that is same as input path but with '_out.mp4'
    path_input_obj = Path(path_input)
    path_tmp = str(path_input_obj.with_stem(path_input_obj.stem + "_tmp"))

    out = cv2.VideoWriter(path_tmp, cv2.VideoWriter_fourcc(*'mp4v'), used_fps, size)
    # out = cv2.VideoWriter(path_tmp, cv2.VideoWriter_fourcc(*'DIVX'), used_fps, size)

    if colorized_frames[0].max() <= 1:
        print("scaling up to 255 range...")
        colorized_frames = [(colorized_frame*255).astype(np.uint8) for colorized_frame in colorized_frames]

    colorized_frames = [np.uint8(image) for image in colorized_frames]

    for colorized_frame in tqdm(colorized_frames):
        # use RGB, not RBG!
        rgb_colorized_frame = cv2.cvtColor(colorized_frame, cv2.COLOR_BGR2RGB)
        out.write(rgb_colorized_frame)
        # out.write(colorized_frame)
    out.release()

    # create output path that is same as input path but with '_out.mp4'
    path_output = str(path_input_obj.with_stem(path_input_obj.stem + "_out"))

    output_video = VideoFileClip(path_tmp)
    output_video = output_video.set_audio(input_audio)
    output_video.write_videofile(path_output, logger=None)

    os.remove(path_tmp)

    print("Done.")
    return path_output
