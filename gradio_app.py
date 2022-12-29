from utils import inference_model

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

def inference(img):
    return inference_model(model, img)

demo = gr.Interface(
    inference, 
    inputs=gr.components.Image(type="pil"), 
    outputs=gr.components.Image(type="pil"),
    title="DCGANs Colorizer",
    description="Demo of Colorization Model",
    allow_flagging='never',
)

demo.launch()
