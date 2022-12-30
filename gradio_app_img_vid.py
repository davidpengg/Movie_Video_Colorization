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

