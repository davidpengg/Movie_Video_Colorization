import torch
from torchvision import transforms

import numpy as np
from skimage.color import rgb2lab, lab2rgb
import skimage.transform
from PIL import Image

from pytube import YouTube

import os
from tqdm import tqdm
from moviepy.editor import AudioFileClip, VideoFileClip
import cv2

from pdb import set_trace

def get_progvid_from_youtube(youtube_url, path_progvid):
    """Creates progvid file from YouTube url.
    """
    yt = YouTube(youtube_url)
    x = yt.streams.filter(progressive=True, file_extension='mp4')
    print(x)
    x[0].download(filename=path_progvid)

def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """
    # set_trace()
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)

SIZE = 256
def inference_model_img(model, img):
    # set_trace()
    # save old size

    if not isinstance(img, Image.Image):
        # set_trace()
        img = Image.fromarray(img)
    og_size = img.size
    # resize img to be SIZE x SIZE
    img = transforms.Resize((SIZE, SIZE),  transforms.InterpolationMode.BICUBIC)(img)

    img = np.array(img)
    img_lab = rgb2lab(img).astype("float32")
    img_lab = transforms.ToTensor()(img_lab)
    L = img_lab[[0], ...] / 50. - 1. # Between -1 and 1
    # ab = img_lab[[1, 2], ...] / 110. # Between -1 and 1

    L = L[None]
    # ab = ab[None]
    # data = {"L": L, "ab": ab}

    #   Load image into model
    # model.setup_input(data)
    model.L = L.to(model.device)
    model.forward()

    fake_color = model.fake_color.detach()
    # real_color = model.ab
    # L = model.L
    fake_imgs = lab_to_rgb(L, fake_color)
    # real_imgs = lab_to_rgb(L, real_color)

    fake_img = fake_imgs[0]

    # get original size
    fake_img = skimage.transform.resize(fake_img, og_size[::-1])

    return fake_img

def inference_model_vid(model, vid: str):
    # iter_frames is an iterator

    input_video = VideoFileClip(vid)

    colorized_frames = []
    nframes = input_video.reader.nframes
    print(f"There are {nframes} frames to process")
    for i, frame in tqdm(enumerate(input_video.iter_frames()), total=nframes):
    
        color_frame = inference_model_img(model, frame)
        colorized_frames.append(color_frame)
    
    height, width = colorized_frames[0].shape[:2]
    size = (width, height)
    set_trace()
    out = cv2.VideoWriter('tmp_output.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 24, size)

    for colorized_frame in colorized_frames:
        scaled_colorized_frame = (colorized_frame*255).astype(np.uint8)
        out.write(scaled_colorized_frame)
    out.release()
    set_trace()

    # add audio too
    output_video = VideoFileClip("tmp_output.mp4")
    output_video = output_video.set_audio(input_video.audio)
    output_video.write_videofile("output.mp4")

    os.remove("tmp_output.mp4")
    return "output.mp4"
