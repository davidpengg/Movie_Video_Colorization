import torch
import gradio as gr
from pytube import YouTube

from pdb import set_trace

from colorizer import colorize_vid
from dcgan import *

# ================================

# EXAMPLE_FPS = "Same as original"
EXAMPLE_FPS = 12
examples = [
    ["examples/1_falcon.mp4", "modelv2", EXAMPLE_FPS],
    ["examples/2_mughal.mp4", "modelv1", EXAMPLE_FPS],
    ["examples/3_wizard.mp4", "modelv1", EXAMPLE_FPS],
    # ["examples/4_elgar.mp4", "modelv2", EXAMPLE_FPS]
]

model_choices = [
    "modelv2",
    "modelv1",
]

loaded_models = {}
for model_weights in model_choices:
    model = torch.load(model_weights, map_location=torch.device('cpu'))
    model.eval()  # also done in colorizer
    loaded_models[model_weights] = model


def colorize_video(path_video, chosen_model, chosen_fps, start='', end=''):
    if not path_video:
        return
    return colorize_vid(
        path_video,
        loaded_models[chosen_model],
        chosen_fps,
        start,
        end
    )


def download_youtube(url):
    try:
        yt = YouTube(url)
        streams = yt.streams.filter(
            progressive=True,
            file_extension='mp4').order_by('resolution')
        return streams[0].download()
    except BaseException:
        raise Exception("Invalid URL or Video Unavailable")


app = gr.Blocks()
with app:
    gr.Markdown("# <p align='center'>Movie and Video Colorization</p>")
    gr.Markdown(
        """
        <p style='text-align: center'>
        Colorize black-and-white movies or videos with a DCGAN-based model!
        <br>
        Project by David Peng, Annie Lin, Adam Zapatka, and Maggy Lambo.
        <p>
        """
    )

    gr.Markdown("### Step 1: Choose a YouTube video (or upload locally below)")

    youtube_url = gr.Textbox(label="YouTube Video URL")

    youtube_url_btn = gr.Button(value="Extract YouTube Video")

    with gr.Row():
        gr.Markdown("### Step 2: Adjust settings")
        gr.Markdown("### Step 3: Hit \"Colorize\"")
    with gr.Row():
        bw_video = gr.Video(label="Black-and-White Video")
        colorized_video = gr.Video(label="Colorized Video")
    with gr.Row():
        with gr.Column():
            with gr.Row():
                start_time = gr.Text(
                    label="Start Time (hh:mm:ss or blank for original)", value='')
                end_time = gr.Text(
                    label="End Time (hh:mm:ss or blank for original)", value='')
        with gr.Column():
            bw_video_btn = gr.Button(value="Colorize", variant="primary")
    with gr.Row():
        with gr.Column():
            model_dropdown = gr.Dropdown(
                model_choices,
                value=model_choices[0],
                label="Model"
            )

            fps_dropdown = gr.Dropdown(
                [3, 6, 12, 24, 30, "Same as original"],
                value=6,
                label="FPS of Colorized Video"
            )

            gr.Markdown(
                """
                #### Colorization Notes
                - Leave start, end times blank to colorize the entire video
                - To lower colorization time, you can decrease FPS, resolution, or duration
                - *modelv2* tends to color videos orange and sepia
                - *modelv1* tends to color videos with a variety of colors
                - *modelv2* and *modelv1* use the same modified DCGAN architecture but differ in results because of randomization in training

                #### More Reading
                - <a href='https://towardsdatascience.com/colorizing-black-white-images-with-u-net-and-conditional-gan-a-tutorial-81b2df111cd8' target='_blank'>Colorizing black & white images with U-Net and conditional GAN</a>
                - <a href='https://arxiv.org/abs/1803.05400' target='_blank'>Image Colorization with Generative Adversarial Networks</a>
                """
            )
        with gr.Column():
            gr.Examples(
                examples=examples,
                inputs=[bw_video, model_dropdown, fps_dropdown],
                outputs=[colorized_video],
                fn=colorize_video,
                # cache_examples=True,
            )

    youtube_url_btn.click(
        download_youtube,
        inputs=youtube_url,
        outputs=bw_video
    )

    bw_video_btn.click(
        colorize_video,
        inputs=[bw_video, model_dropdown, fps_dropdown, start_time, end_time],
        outputs=colorized_video
    )

app.launch()
