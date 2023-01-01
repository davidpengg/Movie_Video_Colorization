import gradio as gr
import torch
from pytube import YouTube
import os

from pdb import set_trace

from colorizer import colorize_vid
from dcgan import *

# ================================

# TODO remove when putting on huggingface
for file in os.listdir():
    if file.endswith(".mp4"):
        os.remove(file)

model_choices = [
    "mymodel9.pth",
    "modelbatch_epoch9.pth",
]

loaded_models = {}
for model_weights in model_choices:
    model = torch.load(model_weights, map_location=torch.device('cpu'))
    model.eval() # also done in colorizer
    loaded_models[model_weights] = model

# will be changed by dropdowns
chosen_model = model_choices[0]
chosen_fps = 6

# will be changed upon video being uploaded
chosen_start = 0
chosen_end = ''

def choose_model(model_dropdown_choice):
    global chosen_model
    chosen_model = model_dropdown_choice

def choose_fps(fps_dropdown_choice):
    global chosen_fps
    chosen_fps = fps_dropdown_choice

def colorize_video(path_video, start, end):
    return colorize_vid(loaded_models[chosen_model], path_video, chosen_fps, start, end)

def download_youtube(url):
    yt = YouTube(url)
    streams = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution')
    return streams[0].download()

app = gr.Blocks()
with app:
    # gr.Markdown("# **<p align='center'>Video Colorizer</p>**")
    # gr.Markdown(
    #     """
    #     <p style='text-align: center'>
    #     Perform video classification with <a href='https://huggingface.co/models?pipeline_tag=video-classification&library=transformers' target='_blank'>HuggingFace Transformers video models</a>.
    #     <br> For zero-shot classification, you can use the <a href='https://huggingface.co/spaces/fcakyon/zero-shot-video-classification' target='_blank'>zero-shot classification demo</a>.
    #     </p>
    #     """
    # )
    # gr.Markdown(
    #     """
    #     <p style='text-align: center'>
    #     Follow me for more! 
    #     <br> <a href='https://twitter.com/fcakyon' target='_blank'>twitter</a> | <a href='https://github.com/fcakyon' target='_blank'>github</a> | <a href='https://www.linkedin.com/in/fcakyon/' target='_blank'>linkedin</a> | <a href='https://fcakyon.medium.com/' target='_blank'>medium</a>
    #     </p>
    #     """
    # )

    # with gr.Row(): 
        # with gr.Tab(label="Local File"):
        #     with gr.Row():
        #         with gr.Column():
        #             local_video = gr.Video(label="Black and White Video")
        #             with gr.Row():
        #                 local_start = gr.Text(label="Start (hh:mm:ss)")
        #                 local_end = gr.Text(label="End (hh:mm:ss)")
        #             local_video_btn = gr.Button(value="Submit")

        #         colorized_local_video = gr.Video(label="Colorized Video")

        # with gr.Tab(label="YouTube"):
        #     with gr.Row():
        #         with gr.Column():
        #             youtube_url = gr.Textbox(label="YouTube Video URL")
        #             youtube_url_btn = gr.Button(value="Download YouTube Video")
        #             online_video = gr.Video(label="Black and White Video")
        #             with gr.Row():
        #                 online_start = gr.Text(label="Start (hh:mm:ss)")
        #                 online_end = gr.Text(label="End (hh:mm:ss)")
        #             online_video_btn = gr.Button(value="Submit to Colorize!")
                
        #         colorized_online_video = gr.Video(label="Colorized Video")
    with gr.Row():
        with gr.Column():
            youtube_url = gr.Textbox(label="YouTube Video URL")
            youtube_url_btn = gr.Button(value="Download YouTube Video")
            online_video = gr.Video(label="Black and White Video")
            with gr.Row():
                online_start = gr.Text(label="Start (hh:mm:ss)", value=chosen_start)
                online_end = gr.Text(label="End (hh:mm:ss)", value=chosen_end)
            online_video_btn = gr.Button(value="Submit to Colorize!")
        
        with gr.Column():
            gr.File(label="dummy", visible=False)
            colorized_online_video = gr.Video(label="Colorized Video")
            gr.File(label="dummy", visible=False)

    with gr.Row():
        with gr.Column():
            model_dropdown = gr.Dropdown(
                model_choices,
                value=model_choices[0],
                label="Model"
            )

            fps_dropdown = gr.Dropdown(
                [2, 6, 12, 24, 30, "Same as original FPS"],
                value=6, 
                label="FPS of Colorized Video"
            )

        # gr.Examples(
        #     [["examples/" + example] for example in os.listdir("examples") if ".mp4" in example],
        #     inputs=[online_video, ],
        #     outputs=colorized_online_video
        # )
    # button and dropdown functions

    model_dropdown.change(choose_model, inputs=model_dropdown)

    fps_dropdown.change(choose_fps, inputs=fps_dropdown)

    # local_video_btn.click(
    #     colorize_video,
    #     inputs=[local_video, local_start, local_end],
    #     outputs=colorized_local_video
    # )

    youtube_url_btn.click(
        download_youtube,
        inputs=youtube_url,
        outputs=online_video
    )

    online_video_btn.click(
        colorize_video,
        inputs=[online_video, online_start, online_end],
        outputs=colorized_online_video
    )


app.launch()
