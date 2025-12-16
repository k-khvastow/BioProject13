from src.models.NCA import SegNCA
from src.losses.LossFunctions import DiceLoss
from src.agents.Agent import Agent

import torch
from torchviz import make_dot

# 1. Instantiate the model
# (It will automatically move to GPU if available because of the __init__ logic)
model = SegNCA(channel_n=16, hidden_size=128)

# 2. Create dummy input
# Shape: [Batch, Height, Width, Channels]
dummy_input = torch.randn(1, 32, 32, 16)

# --- THE FIX ---
# Move the input to the same device as the model (likely 'cuda:0')
dummy_input = dummy_input.to(model.device) 
# ----------------

# 3. Run a SINGLE update step
dummy_output = model.update(dummy_input, fire_rate=0.5)

# 4. Generate the graph
# We use dict(model.named_parameters()) to ensure the graph labels the learned weights
# Change this line in your python script
viz = make_dot(dummy_output, params=dict(model.named_parameters()))

# Instead of .render(), just print the source to verify it works
print(viz.source) 

# Or save the dot file without trying to convert to PNG
viz.save("nca_architecture.dot")