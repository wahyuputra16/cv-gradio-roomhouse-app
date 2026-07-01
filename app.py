import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from torchvision.models import resnet18
import torch
import gradio as gr

from google.colab import files
from PIL import Image



demo = gr.Interface(

    fn=predict,

    inputs=gr.Image(type="pil"),

    outputs=gr.Textbox(),

    title="Indoor Scene Classification",

    description="Upload an indoor image to classify the room type."

)

demo.launch(debug=True)
