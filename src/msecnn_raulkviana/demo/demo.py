import gradio as gr
import cv2 as cv
import sys
sys.path.append("../")
import msecnn
import dataset_utils as du

def pipeline(img):
    # Convert image to appropriate format
    img_rgb = cv.cvtColor(img, cv.COLOR_RGB2BGR)

    # Convert image to YUV
    img_yuv = du.bgr2yuv(img_rgb)

    # Obtain 128 by 128 top left area

    # Pass image by model

    # Obtain best split

    # Modify image to illustrate best split on it

    # Return best split and modified image

with open("description.md", encoding="utf-8") as f:
    description = f.read()

demo = gr.Interface(fn=greet, inputs="image", thumbnail="msecnn_model.png",
                    outputs=["image", "text"], description=description,
                    title="MSE-CNN Demo", image="msecnn_model.png")

demo.launch() 