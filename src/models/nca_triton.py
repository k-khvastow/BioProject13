import torch
import triton
import triton.language as tl
import pdb

@triton.jit
def _nca_full_step_kernel(
    x_ptr,           # Current State (B, H, W, C)
    input_img_ptr,   # Original Input Image
    out_ptr,         # Output State
    
    # Weight Pointers
    p0_ptr, p1_ptr,      
    w0_ptr, b0_ptr, w1_ptr,             
    
    # Scalars
    seed, fire_rate,
    
    # Strides
    stride_b, stride_h, stride_w, stride_c,
    
    # Meta-parameters
    H: tl.constexpr, 
    W: tl.constexpr, 
    C: tl.constexpr, 
    HIDDEN: tl.constexpr, 
    INPUT_CHANNELS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # 1. Coordinate Mapping
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total_pixels = tl.num_programs(0) * BLOCK_SIZE
    mask = idx < total_pixels 
    
    w_idx = idx % W
    h_idx = (idx // W) % H
    b_idx = idx // (H * W)

    # 2. Offsets with EXPLICIT stride usage
    # We multiply by stride_c to ensure the compiler doesn't optimize the argument away.
    # For contiguous tensors, stride_c is 1, so this changes nothing mathematically.
    c_offs = tl.arange(0, C) * stride_c
    h_offs = tl.arange(0, HIDDEN)
    
    # Base pointer for this pixel
    base_offset = b_idx * stride_b + h_idx * stride_h + w_idx * stride_w
    
    # -----------------------------------------------------------
    # A. PERCEPTION & MLP LAYER 0
    # -----------------------------------------------------------
    
    # Weight Pointers
    w0_x_ptr  = w0_ptr
    w0_z1_ptr = w0_ptr + (C * HIDDEN)
    w0_z2_ptr = w0_ptr + (2 * C * HIDDEN)

    # Initialize Accumulator with Bias
    bias = tl.load(b0_ptr + h_offs)
    hidden_acc = tl.zeros((BLOCK_SIZE, HIDDEN), dtype=tl.float32) + bias[None, :]

    # --- 1. Center Pixel ---
    x_center_ptr = x_ptr + base_offset
    # Load (Block, C)
    x_center = tl.load(x_center_ptr[:, None] + c_offs[None, :], mask=mask[:, None])
    
    w0_x = tl.load(w0_x_ptr + c_offs[:, None] * HIDDEN + h_offs[None, :])
    hidden_acc += tl.dot(x_center, w0_x)

    # --- 2. Neighbors ---
    neighbor_idx = 0
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            ny = tl.maximum(0, tl.minimum(H - 1, h_idx + dy))
            nx = tl.maximum(0, tl.minimum(W - 1, w_idx + dx))
            
            neigh_offset = b_idx * stride_b + ny * stride_h + nx * stride_w
            x_neigh_ptr = x_ptr + neigh_offset
            x_neigh = tl.load(x_neigh_ptr[:, None] + c_offs[None, :], mask=mask[:, None])
            
            # Load Depthwise Weights
            p0_val = tl.load(p0_ptr + neighbor_idx * C + c_offs)
            p1_val = tl.load(p1_ptr + neighbor_idx * C + c_offs)
            
            z1 = x_neigh * p0_val[None, :]
            z2 = x_neigh * p1_val[None, :]
            
            w0_z1 = tl.load(w0_z1_ptr + c_offs[:, None] * HIDDEN + h_offs[None, :])
            w0_z2 = tl.load(w0_z2_ptr + c_offs[:, None] * HIDDEN + h_offs[None, :])
            
            hidden_acc += tl.dot(z1, w0_z1)
            hidden_acc += tl.dot(z2, w0_z2)
            
            neighbor_idx += 1

    # -----------------------------------------------------------
    # B. ACTIVATION & UPDATE
    # -----------------------------------------------------------
    
    hidden_act = tl.maximum(hidden_acc, 0.0)
    
    w1 = tl.load(w1_ptr + h_offs[:, None] * C + c_offs[None, :])
    update_vector = tl.dot(hidden_act, w1)
    
    # Stochastic Update
    r = tl.rand(seed, idx) 
    do_update = r > fire_rate
    
    # do_update is (BLOCK,), update_vector is (BLOCK, C). Broadcast works automatically here.
    new_x = x_center + update_vector * do_update[:, None].to(tl.float32)
    
    # -----------------------------------------------------------
    # C. HARD CONSTRAINTS
    # -----------------------------------------------------------
    
    if INPUT_CHANNELS > 0:
        # Create mask (1, C)
        # We assume INPUT_CHANNELS is small (e.g., 3). 
        channel_cond = c_offs < INPUT_CHANNELS
        
        # Load input image
        inp_ptr_base = input_img_ptr + base_offset
        img_vals = tl.load(inp_ptr_base[:, None] + c_offs[None, :], mask=mask[:, None])
        
        # Robust Broadcast for tl.where
        # channel_cond is (C). We need (BLOCK, C). 
        # We use [None, :] to expand dimensions explicitly.
        c_mask = channel_cond[None, :]
        
        new_x = tl.where(c_mask, img_vals, new_x)

    # -----------------------------------------------------------
    # D. STORE
    # -----------------------------------------------------------
    
    final_out_ptr = out_ptr + base_offset[:, None] + c_offs[None, :]
    tl.store(final_out_ptr, new_x, mask=mask[:, None])


def run_nca_step(
    x_in, input_image, x_out,
    w_p0, w_p1,
    w_fc0, b_fc0,
    w_fc1,
    fire_rate, seed
):
    B, H, W, C = x_in.shape
    Hidden = b_fc0.shape[0]
    
    total_pixels = B * H * W
    BLOCK_SIZE = 256
    grid = (triton.cdiv(total_pixels, BLOCK_SIZE),)
    
    stride_b, stride_h, stride_w, stride_c = x_in.stride()

    # We cast to standard python types to avoid JIT type confusion
    seed_val = int(seed)
    fire_rate_val = float(fire_rate)

    _nca_full_step_kernel[grid](
        x_in, input_image, x_out,
        w_p0, w_p1,
        w_fc0, b_fc0,
        w_fc1,
        seed_val, fire_rate_val,
        stride_b, stride_h, stride_w, stride_c,
        H=H, W=W, C=C, HIDDEN=Hidden,
        INPUT_CHANNELS=3, 
        BLOCK_SIZE=BLOCK_SIZE
    )