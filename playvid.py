import cv2
import os
from pytube import YouTube
from moviepy.editor import AudioFileClip, VideoFileClip

from pdb import set_trace

def get_progvid_from_youtube(youtube_url, path_progvid):
    """Creates progvid file from YouTube url.
    """

    yt = YouTube(youtube_url)
    x = yt.streams.filter(progressive=True, file_extension='mp4')
    x[0].download(filename=path_progvid)

def get_audio_from_progvid(path_progvid, path_audio):
    """Creates audio file from progvid path.
    """

    clip = VideoFileClip(path_progvid)
    clip.audio.write_audiofile(path_audio)

def add_audio_to_clip(path_clip, path_audio, path_combine):
    """Creates new video combining clip and audio.
    """

    clip = VideoFileClip(path_clip)
    audio = AudioFileClip(path_audio)
    new_video = clip.set_audio(audio)
    new_video.write_videofile(path_combine)

def clip_to_frames(path_clip, path_frames):
    # make directory if it doesn't exist
    if not os.path.exists(path_frames):
        os.makedirs(path_frames)

    vidcap = cv2.VideoCapture(path_clip)
    success,image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(f"{path_frames}/frame{count}.jpg", image)
        success,image = vidcap.read()
        # print('Read a new frame: ', success)
        count += 1

def frames_to_clip(path_frames, path_clip):
    frame_array = []
    files = [f for f in os.listdir(path_frames) if os.path.isfile(os.path.join(path_frames, f))]
    #for sorting the file names properly
    # files.sort(key = lambda x: int(x[5:-4]))
    files.sort()
    for file in files:
        filename= f"{path_frames}/{file}"
        img = cv2.imread(filename)
        frame_array.append(img)

    # get size
    tmp_img = cv2.imread(f"{path_frames}/{files[0]}")
    height, width, _ = tmp_img.shape
    size = (width,height)
    out = cv2.VideoWriter(path_clip, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for frame in frame_array:
        out.write(frame)
    out.release()

YOUTUBE_OZ = "https://www.youtube.com/watch?v=PSZxmZmBfnU"
if __name__ == "__main__":

    tmpdir = "tmp_files/"
    progvid = f"{tmpdir}/progvid.mp4"
    audio = f"{tmpdir}/audio.wav" # this must be a moviepy-supported filetype
    input_frames = f"{tmpdir}/input_frames"

    # get_progvid_from_youtube(YOUTUBE_OZ, progvid)
    # get_audio_from_progvid(progvid, audio)
    clip_to_frames(progvid, input_frames)


