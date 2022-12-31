from colorizer import colorize_vid

import gradio as gr
import torch
from dcgan import *

import os

from pdb import set_trace

if __name__ == "__main__":
    use_models = [
        "modelbatch_epoch9.pth",
        "mymodel9.pth",
    ]

    # model_weights = "modelbatch_epoch9.pth"
    # model = torch.load(model_weights, map_location=torch.device('cpu'))
    # print(model.net_G)
    # model.eval()

    loaded_models = {}
    for model_weights in use_models:
        model = torch.load(model_weights, map_location=torch.device('cpu'))
        model.eval() # also done in colorizer
        loaded_models[model_weights] = model

    # for _, model in loaded_models.items():
    #     assert(model.training == False)

    INPUT_FPS = "Same as Video"
    def colorize(vid, fps, model_weights):
        if fps == INPUT_FPS:
            return colorize_vid(loaded_models[model_weights], vid)
        else:
            return colorize_vid(loaded_models[model_weights], vid, fps=fps)

    # gr.Slider(minimum=1, maximum=30, value=24, step=1, label="FPS of Colorized Video"),
    # gr.Radio([2, 5, 12, 24, 30], value=5, label="FPS of Colorized Video"),

    demo = gr.Interface(
        colorize, 
        inputs=[
            gr.Video(label="Black and White"),
            gr.Dropdown([2, 6, 12, 24, 30, INPUT_FPS], value=6, label="FPS of Colorized Video"),
            gr.Dropdown(use_models, value=use_models[0], label="Model Weights")
        ], 
        outputs=gr.Video(label="Colorized"),
        examples=[["examples/" + example, 6, use_models[0]] for example in os.listdir("examples")],
        title="Image Colorizer and Video Colorizer", #🎬 🌈
        description="Demo of Video Colorization Model!\nTraining time is affected by 1) length of video, 2) FPS of video, 3) video resolution",
        allow_flagging='never',
        cache_examples=True
    )

    demo.launch()