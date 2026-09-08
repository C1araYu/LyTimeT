import os
import sys
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
from scipy.integrate import solve_ivp
import torch
from torchdiffeq import odeint
import cv2

def engine_gpu(rng, num_frm, fps=60, device='cuda'):
    # Parameters for 4 coupled harmonic oscillators (8th-order system)
    m1, m2, m3, m4 = 1.0, 1.0, 1.0, 1.0     # masses (kg)
    k1, k2, k3, k4 = 10.0, 12.0, 8.0, 15.0  # spring constants (N/m)
    k_c12, k_c23, k_c34 = 5.0, 4.0, 6.0     # coupling spring constants (N/m)
    b1, b2, b3, b4 = 0.1, 0.15, 0.08, 0.12  # damping coefficients

    dt = 1.0 / fps
    t_eval = torch.linspace(0, num_frm*dt, num_frm).to(device)
    
    # Initial state as a torch tensor on the device
    # y = [x1, v1, x2, v2, x3, v3, x4, v4] (8th-order system)
    initial_state = torch.tensor([
        rng.uniform(-2.0, 2.0),  # x1 initial position
        rng.uniform(-3.0, 3.0),  # v1 initial velocity
        rng.uniform(-2.0, 2.0),  # x2 initial position
        rng.uniform(-3.0, 3.0),  # v2 initial velocity
        rng.uniform(-2.0, 2.0),  # x3 initial position
        rng.uniform(-3.0, 3.0),  # v3 initial velocity
        rng.uniform(-2.0, 2.0),  # x4 initial position
        rng.uniform(-3.0, 3.0)   # v4 initial velocity
    ], dtype=torch.float32, device=device)

    # Define the ODE function for 4 coupled harmonic oscillators
    def f(t, y):
        x1, v1, x2, v2, x3, v3, x4, v4 = y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]
        
        # Equations of motion for 4 coupled harmonic oscillators in a chain
        dx1_dt = v1
        dv1_dt = (-k1 * x1 - k_c12 * (x1 - x2) - b1 * v1) / m1
        
        dx2_dt = v2
        dv2_dt = (-k2 * x2 - k_c12 * (x2 - x1) - k_c23 * (x2 - x3) - b2 * v2) / m2
        
        dx3_dt = v3
        dv3_dt = (-k3 * x3 - k_c23 * (x3 - x2) - k_c34 * (x3 - x4) - b3 * v3) / m3
        
        dx4_dt = v4
        dv4_dt = (-k4 * x4 - k_c34 * (x4 - x3) - b4 * v4) / m4
        
        return torch.stack([dx1_dt, dv1_dt, dx2_dt, dv2_dt, dx3_dt, dv3_dt, dx4_dt, dv4_dt])
    
    # Solve the ODE on GPU
    sol = odeint(f, initial_state, t_eval, rtol=1e-6)
    # Move solution back to CPU and convert to numpy
    states = sol.cpu().numpy()
    return states

def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def engine(rng, num_frm, fps=60):
    # Parameters for 4 coupled harmonic oscillators (8th-order system)
    m1, m2, m3, m4 = 1.0, 1.0, 1.0, 1.0     # masses (kg)
    k1, k2, k3, k4 = 10.0, 12.0, 8.0, 15.0  # spring constants (N/m)
    k_c12, k_c23, k_c34 = 5.0, 4.0, 6.0     # coupling spring constants (N/m)
    b1, b2, b3, b4 = 0.1, 0.15, 0.08, 0.12  # damping coefficients

    dt = 1.0 / fps
    t_eval = np.arange(num_frm) * dt
    
    # y = [x1, v1, x2, v2, x3, v3, x4, v4] (8th-order system)
    f = lambda t, y: [
        y[1],  # dx1/dt = v1
        (-k1 * y[0] - k_c12 * (y[0] - y[2]) - b1 * y[1]) / m1,  # dv1/dt
        y[3],  # dx2/dt = v2
        (-k2 * y[2] - k_c12 * (y[2] - y[0]) - k_c23 * (y[2] - y[4]) - b2 * y[3]) / m2,  # dv2/dt
        y[5],  # dx3/dt = v3
        (-k3 * y[4] - k_c23 * (y[4] - y[2]) - k_c34 * (y[4] - y[6]) - b3 * y[5]) / m3,  # dv3/dt
        y[7],  # dx4/dt = v4
        (-k4 * y[6] - k_c34 * (y[6] - y[4]) - b4 * y[7]) / m4   # dv4/dt
    ]
    
    initial_state = [
        rng.uniform(-2.0, 2.0),  # x1 initial position
        rng.uniform(-3.0, 3.0),  # v1 initial velocity
        rng.uniform(-2.0, 2.0),  # x2 initial position
        rng.uniform(-3.0, 3.0),  # v2 initial velocity
        rng.uniform(-2.0, 2.0),  # x3 initial position
        rng.uniform(-3.0, 3.0),  # v3 initial velocity
        rng.uniform(-2.0, 2.0),  # x4 initial position
        rng.uniform(-3.0, 3.0)   # v4 initial velocity
    ]
    
    sol = solve_ivp(
        f, 
        [0.0, num_frm * dt],
        initial_state, 
        t_eval=t_eval, 
        rtol=1e-6
    )
    
    states = sol.y.T
    return states

def render(x1, x2, x3, x4):
    """
    Render the 4 coupled oscillators system
    x1, x2, x3, x4: positions of the four masses
    """
    bg_color = (240, 240, 240)
    mass1_color = (220, 50, 50)    # Red for mass 1
    mass2_color = (50, 50, 220)    # Blue for mass 2
    mass3_color = (50, 220, 50)    # Green for mass 3
    mass4_color = (220, 220, 50)   # Yellow for mass 4
    spring_color = (100, 100, 100)  # Gray for springs
    wall_color = (80, 80, 80)      # Dark gray for walls
    
    img_size = (128, 128)
    im = Image.new('RGB', img_size, bg_color)
    draw = ImageDraw.Draw(im)
    
    # Coordinate system: center at (64, 64), scale positions
    center_y = 64
    scale = 15  # pixels per unit length (smaller scale for 4 masses)
    
    # Wall positions (fixed points)
    wall_left = 5
    wall_right = 123
    
    # Calculate mass positions (constrained to image bounds)
    pos1 = max(10, min(118, 20 + x1 * scale))
    pos2 = max(10, min(118, 40 + x2 * scale))
    pos3 = max(10, min(118, 60 + x3 * scale))
    pos4 = max(10, min(118, 80 + x4 * scale))
    
    # Draw walls
    draw.rectangle([wall_left-2, center_y-40, wall_left, center_y+40], fill=wall_color)
    draw.rectangle([wall_right, center_y-40, wall_right+2, center_y+40], fill=wall_color)
    
    # Draw springs (simplified as lines with small zigzags)
    def draw_spring(start_pos, end_pos, y_offset=0):
        points = []
        for i in range(8):
            x = start_pos + (end_pos - start_pos) * i / 7
            y = center_y + y_offset + (-1)**(i) * 3
            points.append((x, y))
        for i in range(len(points)-1):
            draw.line([points[i], points[i+1]], fill=spring_color, width=1)
    
    # Spring 1: from left wall to mass 1
    draw_spring(wall_left, pos1, -5)
    
    # Coupling springs between masses
    draw_spring(pos1, pos2, 0)
    draw_spring(pos2, pos3, 5)
    draw_spring(pos3, pos4, -5)
    
    # Spring 4: from mass 4 to right wall
    draw_spring(pos4, wall_right, 5)
    
    # Draw masses as circles (smaller radius for 4 masses)
    mass_radius = 6
    draw.ellipse([pos1-mass_radius, center_y-mass_radius-5, 
                  pos1+mass_radius, center_y+mass_radius-5], fill=mass1_color)
    draw.ellipse([pos2-mass_radius, center_y-mass_radius, 
                  pos2+mass_radius, center_y+mass_radius], fill=mass2_color)
    draw.ellipse([pos3-mass_radius, center_y-mass_radius+5, 
                  pos3+mass_radius, center_y+mass_radius+5], fill=mass3_color)
    draw.ellipse([pos4-mass_radius, center_y-mass_radius-5, 
                  pos4+mass_radius, center_y+mass_radius-5], fill=mass4_color)
    
    # Add labels (smaller text)
    draw.text((pos1-3, center_y+10), "1", fill=(0,0,0))
    draw.text((pos2-3, center_y+15), "2", fill=(0,0,0))
    draw.text((pos3-3, center_y+20), "3", fill=(0,0,0))
    draw.text((pos4-3, center_y+10), "4", fill=(0,0,0))
    
    return im

def gen_data(data_filepath, num_seq, num_frm, seed=0):
    rng = np.random.default_rng(seed)
    states = np.zeros((num_seq, num_frm, 8))  # 8 state variables

    os.makedirs(data_filepath, exist_ok=True)

    for n in tqdm(range(num_seq)):
        seq_filepath = os.path.join(data_filepath, str(n))
        mkdir(seq_filepath)
        states[n, :, :] = engine(rng, num_frm)
        for k in range(num_frm):
            # Pass positions of all 4 masses: x1, x2, x3, x4
            im = render(states[n, k, 0], states[n, k, 2], states[n, k, 4], states[n, k, 6])
            im.save(os.path.join(seq_filepath, str(k)+'.png'))
    
    states_dir = os.path.dirname(data_filepath)
    np.save(os.path.join(states_dir, 'states.npy'), states)

def create_demo_video(data_filepath, output_dir='./videos'):
    """Create a demonstration video from one sequence"""
    try:
        import cv2
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Use first sequence for demo
        demo_seq_path = os.path.join(data_filepath, '0')
        if not os.path.exists(demo_seq_path):
            print("No sequence data found for demo video")
            return
            
        # Get all frame files
        frame_files = sorted([f for f in os.listdir(demo_seq_path) if f.endswith('.png')], 
                           key=lambda x: int(x.split('.')[0]))
        
        if len(frame_files) == 0:
            print("No frames found for demo video")
            return
            
        # Load first frame to get dimensions
        first_frame = Image.open(os.path.join(demo_seq_path, frame_files[0]))
        width, height = first_frame.size
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = os.path.join(output_dir, 'coupled_oscillators_demo.mp4')
        video = cv2.VideoWriter(video_path, fourcc, 15.0, (width, height))
        
        # Add frames to video
        for frame_file in frame_files:
            img = Image.open(os.path.join(demo_seq_path, frame_file))
            frame = np.array(img)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video.write(frame_bgr)
        
        video.release()
        print(f"Demo video saved: {video_path}")
        
    except ImportError:
        print("OpenCV not available, skipping demo video generation")

if __name__ == '__main__':
    data_filepath = '/home/kuaiyu/tide/data/physics_prediction_v2/data/coupled_oscillators'
    gen_data(data_filepath, num_seq=1200, num_frm=60)
    
    # Generate demo video after data creation
    create_demo_video(data_filepath)