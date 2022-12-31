import gradio as gr
import torch
from pytube import YouTube
import os

from pdb import set_trace

from colorizer import colorize_vid
from dcgan import *

# ================================

def get_video(url) -> str:
    try:
        yt = YouTube(url)
        # get streams by resolution increasing
        streams = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution')
        if not streams:
            return ''
        print(streams)
        return streams[0].download()
    except Exception as e:
        print(e)
        return ''

model_choices = [
    "modelbatch_epoch9.pth",
    "mymodel9.pth",
]

loaded_models = {}
for model_weights in model_choices:
    model = torch.load(model_weights, map_location=torch.device('cpu'))
    model.eval() # also done in colorizer
    loaded_models[model_weights] = model

# will be changed by dropdowns
chosen_model = model_choices[0]
chosen_fps = 6

def choose_model(model_dropdown_choice):
    global chosen_model
    chosen_model = model_dropdown_choice

def choose_fps(fps_dropdown_choice):
    global chosen_fps
    chosen_fps = fps_dropdown_choice

def colorize_local_video(path_video):
    return colorize_vid(loaded_models[chosen_model], path_video, chosen_fps)

def colorize_online_video(youtube_url):
    path_video = get_video(youtube_url)
    if not path_video:
        # YouTube video grabbing unsuccessful
        return
    path_colorized = colorize_local_video(path_video)
    os.remove(path_video)
    return path_colorized

app = gr.Blocks()
with app:
    gr.Markdown("# **<p align='center'>Video Colorizer</p>**")
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

    with gr.Row(): 
        with gr.Tab(label="Local File"):
            with gr.Row():
                with gr.Column():
                    local_video = gr.Video(label="Black and White Video")
                    local_video_btn = gr.Button(value="Submit")
     
                colorized_local_video = gr.Video(label="Colorized Video")
     
                # gr.Interface(
                #     colorize_local_video,
                #     inputs=gr.Video(label="Black and White Video"),
                #     outputs=gr.Video(label="Colorized Video"),

                # )
        with gr.Tab(label="Youtube"):
            with gr.Row():
                with gr.Column():
                    online_video = gr.Textbox(label="YouTube Video URL")
                    online_video_btn = gr.Button(value="Submit")
                
                colorized_online_video = gr.Video(label="Colorized Video")

                # gr.Interface(
                #     colorize_online_video,
                #     inputs=gr.Video(label="Black and White Video"),
                #     outputs=gr.Video(label="Colorized Video"),
                # )

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
        gr.Textbox(label="dummy", visible=False)

    # button and dropdown functions

    model_dropdown.change(choose_model, inputs=model_dropdown)

    fps_dropdown.change(choose_fps, inputs=fps_dropdown)

    # local_video_btn.click(
    #     colorize_local_video,
    #     inputs=local_video,
    #     outputs=colorized_local_video
    # )

    # online_video_btn.click(
    #     colorize_online_video,
    #     inputs=online_video,
    #     outputs=colorized_online_video
    # )

    # examples 

app.launch()
