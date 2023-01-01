import torch
from torchvision import transforms

import numpy as np
from skimage.color import rgb2lab, lab2rgb
import skimage.transform
from PIL import Image

import os
from tqdm import tqdm
from moviepy.editor import VideoFileClip, AudioFileClip
from moviepy.tools import cvsecs
import cv2

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
    img = transforms.Resize(
        (SIZE, SIZE), transforms.InterpolationMode.BICUBIC)(img)
    img = np.array(img)
    img_lab = rgb2lab(img).astype("float32")
    img_lab = transforms.ToTensor()(img_lab)
    L = img_lab[[0], ...] / 50. - 1.  # Between -1 and 1

    return L


def get_predictions(model, L):
    # model.L = L.to(model.device)
    model.eval()
    with torch.no_grad():
        model.L = L.to(torch.device('cpu'))
        model.forward()
    fake_color = model.fake_color.detach()
    fake_imgs = lab_to_rgb(L, fake_color)

    return fake_imgs


def colorize_img(model, img):
    L = get_L(img)
    L = L[None]  # put in list
    fake_imgs = get_predictions(model, L)
    fake_img = fake_imgs[0]  # get out of list
    resized_fake_img = skimage.transform.resize(
        fake_img, img.size[::-1])  # reshape to original size

    return resized_fake_img


def valid_start_end(duration, start_input, end_input):
    start = start_input
    end = end_input
    if start == '':
        start = 0
    if end == '':
        end = duration

    try:
        start = cvsecs(start)
        end = cvsecs(end)
    except BaseException:
        # start, end aren't actual time values.
        raise Exception("Invalid start, end values")

    # make it minimal maximum length
    start = max(start, 0)
    end = min(duration, end)

    # start must be less than end
    if start >= end:
        raise Exception("Start must be before end.")

    return start, end


def colorize_vid(path_input, model, fps, start_input, end_input):

    original_video = VideoFileClip(path_input)

    # validate start, end
    start, end = valid_start_end(
        original_video.duration, start_input, end_input)

    input_video = original_video.subclip(start, end)

    if isinstance(fps, int):
        used_fps = fps
        nframes = np.round(fps * input_video.duration)
    else:
        used_fps = input_video.fps
        nframes = input_video.reader.nframes
    print(
        f"Colorizing output with FPS: {fps}, nframes: {nframes}, resolution: {input_video.size}.")

    frames = input_video.iter_frames(fps=used_fps)

    # create tmp path that is same as input path but with '_tmp.[suffix]'
    base_path, suffix = os.path.splitext(path_input)
    path_video_tmp = base_path + "_tmp" + suffix

    # create video writer for output
    size = input_video.size
    out = cv2.VideoWriter(
        path_video_tmp,
        cv2.VideoWriter_fourcc(
            *'mp4v'),
        used_fps,
        size)
    # out = cv2.VideoWriter(path_video_tmp, cv2.VideoWriter_fourcc(*'DIVX'), used_fps, size)

    for frame in tqdm(frames, total=nframes):
        # get colorized frame
        color_frame = colorize_img(model, Image.fromarray(frame))

        if color_frame.max() <= 1:
            color_frame = (color_frame * 255).astype(np.uint8)

        color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
        out.write(color_frame)
    out.release()

    # create output path that is same as input path but with '_out.[suffix]'
    path_output = base_path + "_out" + suffix

    # for some reason, subclip doesn't save audio. so make tmp audio file
    path_audio_tmp = base_path + "audio_tmp.mp3"
    input_video.audio.write_audiofile(path_audio_tmp, logger=None)
    input_audio = AudioFileClip(path_audio_tmp)

    output_video = VideoFileClip(path_video_tmp)
    output_video = output_video.set_audio(input_audio)
    output_video.write_videofile(path_output, logger=None)

    os.remove(path_video_tmp)
    os.remove(path_audio_tmp)

    print("Done.")
    return path_output
