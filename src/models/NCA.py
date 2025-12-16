#contains all NCA models

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
 

class MaxNCA(nn.Module):
    #Classification model NaxNCA
    def __init__(self, channel_n=16, fire_rate=0.5, device="cpu", hidden_size=128, input_channels=3, init_method="standard"):
        super(MaxNCA, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.channel_n = channel_n
        self.input_channels = input_channels

        self.p0 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, groups = channel_n, padding_mode="reflect")
        self.p1 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, groups = channel_n, padding_mode="reflect")

        self.fc0 = nn.Linear(channel_n*3, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)

        self.fc2 = nn.Linear(channel_n,128)
        self.fc3 = nn.Linear(128,13)

        with torch.no_grad():
            self.fc1.weight.zero_()

        if init_method == "xavier":
            torch.nn.init.xavier_uniform(self.fc0.weight)
            torch.nn.init.xavier_uniform(self.fc1.weight)

        self.fire_rate = fire_rate
        self.to(self.device)

    def perceive(self, x):
        #perceptive function, outputs perception vector
        z1 = self.p0(x)
        z2 = self.p1(x)
        y = torch.cat((x,z1,z2),1)
        return y

    def update(self, x_in, fire_rate):
        #Update function
        x = x_in.transpose(1,3)

        dx = self.perceive(x)
        dx = dx.transpose(1,3)
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate = self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2),1])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,3)
        x = x.transpose(1,3)

        return x

    def forward(self, x, steps=32, fire_rate=0.5):
        #Forward function, applies k NCA update steps leaving input channels unchanged
        for step in range(steps):
            x2 = self.update(x, fire_rate).clone()
            x = torch.concat((x[...,:self.input_channels], x2[...,self.input_channels:]), 3)
        max=F.adaptive_max_pool2d(x.permute(0, 3, 1, 2), (1, 1))
        max = max.view(max.size(0), -1)
        out=self.fc2(max)
        out = F.relu(out)
        out =self.fc3(out)
        
        return out,x


class SegNCA(nn.Module):
    #Segmentation model
    def __init__(self, channel_n=16, fire_rate=0.5, device="cpu", hidden_size=128, input_channels=3, init_method="standard"):
        super(SegNCA, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.channel_n = channel_n
        self.input_channels = input_channels

        self.p0 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, groups = channel_n, padding_mode="reflect")
        self.p1 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, groups = channel_n, padding_mode="reflect")

        self.fc0 = nn.Linear(channel_n*3, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
        with torch.no_grad():
            self.fc1.weight.zero_()

        if init_method == "xavier":
            torch.nn.init.xavier_uniform(self.fc0.weight)
            torch.nn.init.xavier_uniform(self.fc1.weight)

        self.fire_rate = fire_rate
        self.to(self.device)

    def perceive(self, x):
        z1 = self.p0(x)
        z2 = self.p1(x)
        y = torch.cat((x,z1,z2),1)
        return y

    def update(self, x_in, fire_rate):
        x = x_in.transpose(1,3)

        dx = self.perceive(x)
        dx = dx.transpose(1,3)
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate = self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2),1])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,3)

        x = x.transpose(1,3)

        return x

    def forward(self, x: torch.Tensor, steps: int = 64, fire_rate: float = 0.5) -> torch.Tensor:
        for step in range(steps):
            x2 = self.update(x, fire_rate).clone()
            x = torch.concat((x[...,:self.input_channels], x2[...,self.input_channels:]), 3)
        out=x[...,3]
        
        return out,x

class FastSegNCA(nn.Module):
    """
    Optimized Segmentation NCA (FastSegNCA).
    
    Improvements over original SegNCA:
    1. Memory Layout: Uses (B, C, H, W) natively. Avoids expensive transpose(1,3) calls.
    2. Compute: Replaces nn.Linear with nn.Conv2d(kernel_size=1) for faster processing.
    3. In-Place Operations: Optimizes the channel locking step to avoid extra concatenations.
    """
    def __init__(self, channel_n=16, fire_rate=0.5, device="cpu", hidden_size=128, input_channels=3, init_method="standard"):
        super(FastSegNCA, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.channel_n = channel_n
        self.input_channels = input_channels
        self.fire_rate = fire_rate

        # Perception Layers: Depthwise Convolutions (unchanged, but operate on NCHW)
        self.p0 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, groups=channel_n, padding_mode="reflect")
        self.p1 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, groups=channel_n, padding_mode="reflect")

        # Processing Layers: 1x1 Convolutions (Replacing Linear)
        # Old: Linear(channel_n*3, hidden_size) -> New: Conv2d(..., kernel_size=1)
        self.fc0 = nn.Conv2d(channel_n * 3, hidden_size, kernel_size=1)
        
        # Old: Linear(hidden_size, channel_n, bias=False) -> New: Conv2d(..., kernel_size=1, bias=False)
        self.fc1 = nn.Conv2d(hidden_size, channel_n, kernel_size=1, bias=False)

        # Initialization
        with torch.no_grad():
            self.fc1.weight.zero_()
            
        if init_method == "xavier":
            torch.nn.init.xavier_uniform_(self.fc0.weight)
            torch.nn.init.xavier_uniform_(self.fc1.weight)

        self.to(self.device)

    def perceive(self, x):
        z1 = self.p0(x)
        z2 = self.p1(x)
        y = torch.cat((x, z1, z2), 1)
        return y

    def update(self, x, fire_rate):
        # x is already (Batch, Channel, Height, Width)
        
        # 1. Perceive
        y = self.perceive(x) # Returns (B, 3*C, H, W)
        
        # 2. Process (MLP via 1x1 Convs)
        dx = self.fc0(y)
        dx = F.relu(dx)
        dx = self.fc1(dx) # Returns (B, C, H, W)

        # 3. Stochastic Update
        if fire_rate is None:
            fire_rate = self.fire_rate
            
        # Generate mask directly in (B, 1, H, W) to broadcast across channels
        rand_mask = torch.rand((x.shape[0], 1, x.shape[2], x.shape[3]), device=self.device) > fire_rate
        dx = dx * rand_mask.float()

        # 4. Update State
        return x + dx

    def forward(self, x, steps=32, fire_rate=0.5):
        # Input x is assumed to be (B, C, H, W)
        # If input is (B, H, W, C), you must permute it before passing to this model.
        
        # Capture the original input image (Channels 0-2) to enforce hard constraints
        # This matches the behavior of SegNCA where input channels are reset every step.
        input_image = x[:, :self.input_channels, :, :].clone()

        for step in range(steps):
            x = self.update(x, fire_rate)
            
            # Hard Constraint: Reset the visible channels to the original input image
            # Matches: x = torch.concat((x[...,:3], x2[...,3:]), 3) logic from original code
            x[:, :self.input_channels, :, :] = input_image

        # Extract output (Channel 3 is the mask in SegNCA)
        out = x[:, 3, :, :] # Returns (B, H, W)
        
        return out, x


class SimpleNCA(nn.Module):
    #Classification Model SimpleNCA
    def __init__(self, channel_n=16, fire_rate=0.5, device="cpu", hidden_size=128, input_channels=3, init_method="standard"):
        super(SimpleNCA, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.channel_n = channel_n
        self.input_channels = input_channels

        self.p0 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, groups = channel_n, padding_mode="reflect")
        self.p1 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, groups = channel_n, padding_mode="reflect")

        self.fc0 = nn.Linear(channel_n*3, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)

        with torch.no_grad():
            self.fc1.weight.zero_()

        if init_method == "xavier":
            torch.nn.init.xavier_uniform(self.fc0.weight)
            torch.nn.init.xavier_uniform(self.fc1.weight)

        self.fire_rate = fire_rate
        self.to(self.device)

    def perceive(self, x):
        z1 = self.p0(x)
        z2 = self.p1(x)
        y = torch.cat((x,z1,z2),1)
        return y

    def update(self, x_in, fire_rate):
        x = x_in.transpose(1,3)

        dx = self.perceive(x)
        dx = dx.transpose(1,3)
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate = self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2),1])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,3)
        x = x.transpose(1,3)

        return x

    def forward(self, x, steps=32, fire_rate=0.5):
        for step in range(steps):
            x2 = self.update(x, fire_rate).clone()
            x = torch.concat((x[...,:self.input_channels], x2[...,self.input_channels:]), 3)
        out = x.mean([1,2])
        out=out[...,self.input_channels:self.input_channels+13]
        
        return out,x
    

class ConvNCA(nn.Module):
    # Classification Model ConvNCA
    def __init__(self, channel_n=16, fire_rate=0.5, device="cpu", hidden_size=128, input_channels=3, init_method="standard"):
        super(ConvNCA, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.channel_n = channel_n
        self.input_channels = input_channels

        self.p0 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, groups = channel_n, padding_mode="reflect")
        self.p1 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, groups = channel_n, padding_mode="reflect")

        self.fc0 = nn.Linear(channel_n*3, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)

        self.fc2 = nn.Linear(channel_n,128)
        self.fc3 = nn.Linear(128,13)
        
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc2 = nn.Linear(256 * 4 * 4, 512)
        self.fc3 = nn.Linear(512, 13)

        with torch.no_grad():
            self.fc1.weight.zero_()

        if init_method == "xavier":
            torch.nn.init.xavier_uniform(self.fc0.weight)
            torch.nn.init.xavier_uniform(self.fc1.weight)

        self.fire_rate = fire_rate
        self.to(self.device)

    def perceive(self, x):
        z1 = self.p0(x)
        z2 = self.p1(x)
        y = torch.cat((x,z1,z2),1)
        return y

    def update(self, x_in, fire_rate):
        x = x_in.transpose(1,3)

        dx = self.perceive(x)
        dx = dx.transpose(1,3)
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate = self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2),1])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,3)
        x = x.transpose(1,3)

        return x

    def forward(self, x, steps=32, fire_rate=0.5):
        for step in range(steps):
            x2 = self.update(x, fire_rate).clone()
            x = torch.concat((x[...,:self.input_channels], x2[...,self.input_channels:]), 3)
        feature_map=x
        x=x.permute(0, 3, 1, 2)
        x=self.pool(x)
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x))) 
        x = self.pool(F.relu(self.conv3(x)))
        x=torch.flatten(x,1,-1)
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out,feature_map












