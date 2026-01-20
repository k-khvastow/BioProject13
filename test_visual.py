
import torch
import numpy as np
from ncalab.visualization import VisualBinaryImageSegmentation

# 1. Prepare your data
# image: [H, W] or [C, H, W]
# label: [H, W] ground truth
# prediction: The output of your model.forward() or model.record()

# 2. Instantiate the visualizer
visualizer = VisualBinaryImageSegmentation()

# 3. Generate the figure
# prediction_obj needs to be a ncalab.prediction.Prediction instance
fig = visualizer.show(
    model=my_nca_model, 
    image=image_np, 
    prediction=prediction_obj, 
    label=label_np
)

fig.show()

"""
from ncalab.visualization import Animator

# This will create an animation of the segmentation forming
animator = Animator(
    nca=my_nca_model, 
    seed=input_tensor, 
    steps=100,      # number of iterations
    overlay=True    # overlays the mask on the original image
)
"""