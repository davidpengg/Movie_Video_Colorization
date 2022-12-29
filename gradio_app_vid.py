from utils import inference_model_vid

import gradio as gr
import torch
from colorizer_net import *

from pdb import set_trace

with torch.no_grad():
    model = MainModel()
    # model = torch.load("modelbatchv2.pth", map_location=device)
    model.load_state_dict(
        torch.load(
            "modelbatchv2.pth", 
            map_location=torch.device('cpu')
        ).state_dict()
    )
    assert model.device.type == "cpu"
    model.eval()

def inference(vid):

    return inference_model_vid(model, vid)

demo = gr.Interface(
    inference, 
    inputs=gr.Video(), 
    outputs=gr.Video(),
    title="DCGANs Video Colorizer",
    description="Demo of Video Colorization Model",
    allow_flagging='never',
)

demo.launch()