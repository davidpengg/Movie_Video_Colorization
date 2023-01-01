import torch
import os
import gradio as gr
from pytube import YouTube

from pdb import set_trace

from colorizer import colorize_vid
from dcgan import *

# ================================

# TODO remove when putting on huggingface
for file in os.listdir():
    if file.endswith(".mp4"):
        os.remove(file)

model_choices = [
    "modelv2",
    "modelv1",
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

def colorize_video(path_video, start, end):
    if not path_video:
        return
    return colorize_vid(loaded_models[chosen_model], path_video, chosen_fps, start, end)

def download_youtube(url):
    try:
        yt = YouTube(url)
        streams = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution')
        return streams[0].download()
    except:
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

    # v2

    gr.Markdown("### Step 1: Choose a YouTube video (or upload locally below)")

    youtube_url = gr.Textbox(label="YouTube Video URL")
 
    youtube_url_btn = gr.Button(value="Extract YouTube Video")

    with gr.Row().style(equal_height=False):
        with gr.Column():
            gr.Markdown("### Step 2: Adjust settings")

            bw_video = gr.Video(label="Black and White Video")

            with gr.Row():
                start_time = gr.Text(label="Start Time (hh:mm:ss)", value='')
                end_time = gr.Text(label="End Time (hh:mm:ss)", value='')
        
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

        with gr.Column():
            gr.Markdown("### Step 3: Hit \"Colorize\"")

            colorized_video = gr.Video(label="Colorized Video")

            bw_video_btn = gr.Button(value="Colorize", variant="primary")
    # v1
    # with gr.Row():
    #     with gr.Column():
    #         youtube_url = gr.Textbox(label="YouTube Video URL")
    #         youtube_url_btn = gr.Button(value="Download YouTube Video")

    #     gr.Textbox(label="dummy", visible=False)

    # with gr.Row():
    #     with gr.Column():
    #         bw_video = gr.Video(label="Black and White Video")

    #     with gr.Column():
    #         colorized_video = gr.Video(label="Colorized Video")

    # with gr.Row():
    #     with gr.Column():
    #         with gr.Row():
    #             start_time = gr.Text(label="Start (hh:mm:ss)", value=chosen_start)
    #             end_time = gr.Text(label="End (hh:mm:ss)", value=chosen_end)
            
    #         bw_video_btn = gr.Button(value="Submit to Colorize!")

    #         model_dropdown = gr.Dropdown(
    #             model_choices,
    #             value=model_choices[0],
    #             label="Model"
    #         )

    #         fps_dropdown = gr.Dropdown(
    #             [2, 6, 12, 24, 30, "Same as original FPS"],
    #             value=6, 
    #             label="FPS of Colorized Video"
    #         )

    #     gr.Textbox(label="dummy", visible=False)

    # v0
    # with gr.Row():
    #     with gr.Column():
    #         youtube_url = gr.Textbox(label="YouTube Video URL")
    #         youtube_url_btn = gr.Button(value="Download YouTube Video")
    #         bw_video = gr.Video(label="Black and White Video")
    #         with gr.Row():
    #             start_time = gr.Text(label="Start (hh:mm:ss)", value=chosen_start)
    #             end_time = gr.Text(label="End (hh:mm:ss)", value=chosen_end)
    #         bw_video_btn = gr.Button(value="Submit to Colorize!")
        
    #     with gr.Column():
    #         gr.File(label="dummy", visible=False)
    #         colorized_video = gr.Video(label="Colorized Video")
    #         gr.File(label="dummy", visible=False)

    # with gr.Row():
    #     with gr.Column():
    #         model_dropdown = gr.Dropdown(
    #             model_choices,
    #             value=model_choices[0],
    #             label="Model"
    #         )

    #         fps_dropdown = gr.Dropdown(
    #             [2, 6, 12, 24, 30, "Same as original FPS"],
    #             value=6, 
    #             label="FPS of Colorized Video"
    #         )

        # gr.Examples(
        #     [["examples/" + example] for example in os.listdir("examples") if ".mp4" in example],
        #     inputs=[online_video, ], #TODO update these
        #     outputs=colorized_online_video
        # )

    # button and dropdown functions

    model_dropdown.change(choose_model, inputs=model_dropdown)

    fps_dropdown.change(choose_fps, inputs=fps_dropdown)

    youtube_url_btn.click(
        download_youtube,
        inputs=youtube_url,
        outputs=bw_video
    )

    bw_video_btn.click(
        colorize_video,
        inputs=[bw_video, start_time, end_time],
        outputs=colorized_video
    )

app.launch()
