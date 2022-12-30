from utils import inference_model_vid

import gradio as gr
import torch
from colorizer_net import *

from pdb import set_trace

with torch.no_grad():
    # model = MainModel()
    # model.load_state_dict(
    #     torch.load(
    #         "modelbatchv2.pth", 
    #         map_location=torch.device('cpu')
    #     ).state_dict()
    # )
    # assert model.device.type == "cpu"

    model = torch.load("modelbatch_epoch9.pth", map_location=torch.device('cpu'))

    model.eval()

def inference(vid, fps):
    return inference_model_vid(model, vid)

# gr.Slider(minimum=1, maximum=30, value=24, step=1, label="FPS of Colorized Video"),
# gr.Radio([2, 5, 12, 24, 30], value=5, label="FPS of Colorized Video"),
demo = gr.Interface(
    inference, 
    inputs=[
        gr.Video(label="Black and White"),
        gr.Dropdown([2, 6, 12, 24, 30], value=6, label="FPS of Colorized Video"),
    ], 
    outputs=gr.Video(label="Colorized"),
    title="DCGANs Video Colorizer",
    description="Demo of Video Colorization Model",
    allow_flagging='never',
)

demo.launch()