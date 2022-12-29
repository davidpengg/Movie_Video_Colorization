import streamlit as st

import torch
from torchvision import transforms

import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
import skimage.transform

from colorizer_net import *

from pdb import set_trace
# =========================

st.title('DCGANs Colorizer')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache
def load_model():
    print("loading model")
    with torch.no_grad():
        m = MainModel()
        # m = torch.load("modelbatchv2.pth", map_location=device)
        m.load_state_dict(torch.load("modelbatchv2.pth", map_location=torch.device('cpu')).state_dict())
        assert m.device.type == "cpu"
        m.eval()
        return m

SIZE = 256

def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """
    
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)


model = load_model()

file_up = st.file_uploader("Upload an image", type="jpg", label_visibility="hidden")
col1, col2 = st.columns(2)

if file_up is not None:

    #   Preprocess image
    img = Image.open(file_up).convert("RGB")
    
    col1.write("Original Image")
    col1.image(img)
    
    # save old size
    og_size = img.size
    # resize img to be SIZE x SIZE
    img = transforms.Resize((SIZE, SIZE),  transforms.InterpolationMode.BICUBIC)(img)

    img = np.array(img)
    img_lab = rgb2lab(img).astype("float32")
    img_lab = transforms.ToTensor()(img_lab)
    L = img_lab[[0], ...] / 50. - 1. # Between -1 and 1
    ab = img_lab[[1, 2], ...] / 110. # Between -1 and 1

    L = L[None]
    ab = ab[None]
    data = {"L": L, "ab": ab}

    #   Load image into model
    model.setup_input(data)
    model.forward()

    fake_color = model.fake_color.detach()
    real_color = model.ab
    # L = model.L
    fake_imgs = lab_to_rgb(L, fake_color)
    real_imgs = lab_to_rgb(L, real_color)

    fake_img = fake_imgs[0]

    # get original size
    fake_img = skimage.transform.resize(fake_img, og_size[::-1])

    col2.write("Colorized Image")
    col2.image(fake_img)
