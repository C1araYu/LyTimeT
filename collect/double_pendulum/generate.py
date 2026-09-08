import os
import sys
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
from scipy.integrate import solve_ivp
import torch
from torchdiffeq import odeint

def engine_gpu(rng, num_frm, fps=60, device='cuda'):
    # Parameters
    L1 = 0.5     # pendulum 1 length (m)
    L2 = 0.5     # pendulum 2 length (m)
    m1 = 1.0     # pendulum 1 mass (kg)
    m2 = 1.0     # pendulum 2 mass (kg)
    g = 9.81     # gravitational acceleration (m/s^2)

    dt = 1.0 / fps
    t_eval = torch.linspace(0, num_frm*dt, num_frm).to(device)
    
    # Initial state as a torch tensor on the device
    # y = [theta_1, omega_1, theta_2, omega_2]
    initial_state = torch.tensor([
        rng.uniform(-torch.pi, torch.pi),
        rng.uniform(-6, 6),
        rng.uniform(-torch.pi, torch.pi),
        rng.uniform(-10, 10)
    ], dtype=torch.float32, device=device)

    # Define the ODE function using torch operations
    def f(t, y):
        theta1, omega1, theta2, omega2 = y[0], y[1], y[2], y[3]
        sin = torch.sin
        cos = torch.cos
        dtheta1 = omega1
        domega1 = (m2 * g * sin(theta2) * cos(theta1 - theta2) -
                   m2 * sin(theta1 - theta2) * (L1 * (omega1**2) * cos(theta1 - theta2) + L2 * (omega2**2)) -
                   (m1+m2) * g * sin(theta1)) / (L1 * (m1 + m2 * (torch.sin(theta1-theta2)**2))
                  )
        dtheta2 = omega2
        domega2 = (((m1+m2) * (L1 * (omega1**2) * sin(theta1 - theta2) - g * sin(theta2) +
                   g * sin(theta1) * cos(theta1 - theta2)) +
                   m2 * L2 * (omega2**2) * sin(theta1 - theta2) * cos(theta1 - theta2))
                   / (L2 * (m1 + m2 * (torch.sin(theta1-theta2)**2)))
                  )
        return torch.stack([dtheta1, domega1, dtheta2, domega2])
    
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
    L1 = 0.5     # pendulum 1 length (m)
    L2 = 0.5     # pendulum 2 length (m)
    m1 = 1.0     # pendulum 1 mass (kg)
    m2 = 1.0     # pendulum 2 mass (kg)
    g = 9.81     # gravitational acceleration (m/s^2)

    dt = 1.0 / fps
    t_eval = np.arange(num_frm) * dt
    
    # y = [theta_1, omega_1, theta_2, omega_2]
    f = lambda t, y: [
        y[1],
        (m2 * g * np.sin(y[2]) * np.cos(y[0]-y[2]) - m2 * np.sin(y[0]-y[2]) * (L1 * (y[1]**2) * np.cos(y[0]-y[2]) + 
        L2 * (y[3]**2)) - (m1+m2) * g * np.sin(y[0])) / L1 / (m1 + m2 * (np.sin(y[0]-y[2])**2)),
        y[3],
        ((m1+m2) * (L1 * (y[1]**2) * np.sin(y[0]-y[2]) - g * np.sin(y[2]) + g * np.sin(y[0]) * np.cos(y[0]-y[2])) + 
        m2 * L2 * (y[2]**2) * np.sin(y[0]-y[2]) * np.cos(y[0]-y[2])) / L2 / (m1 + m2 * np.sin(y[0]-y[2])**2)
    ]
    initial_state = [
        rng.uniform(-np.pi, np.pi),
        rng.uniform(-6, 6),
        rng.uniform(-np.pi, np.pi),
        rng.uniform(-10, 10)
    ]
    sol = solve_ivp(
        f, 
        [0.0, 1.0],
        initial_state, 
        t_eval=t_eval, 
        rtol=1e-6
    )
    
    states = sol.y.T
    return states
    
def draw_rect(im, col, top_x, top_y, w, h, theta):
    x1 = top_x - w * np.cos(theta) / 2
    y1 = top_y + w * np.sin(theta) / 2
    x2 = x1 + w * np.cos(theta)
    y2 = y1 - w * np.sin(theta)
    x3 = x2 + h * np.sin(theta)
    y3 = y2 + h * np.cos(theta)
    x4 = x3 - w * np.cos(theta)
    y4 = y3 + w * np.sin(theta)
    pts = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

    draw = ImageDraw.Draw(im)
    draw.polygon(pts, fill=col)

def render(theta_1, theta_2):
    bg_color = (215, 205, 192)
    pd1_color = (63, 66, 85)
    pd2_color = (17, 93, 234)
    img_size = (1600, 1600)

    im = Image.new('RGB', img_size, bg_color)
    center = (800, 800)
    
    w1, w2, h1, h2 = 90, 90, 300, 250
    pd1_end_x = center[0] + (h1-35) * np.sin(theta_1)
    pd1_end_y = center[1] + (h1-35) * np.cos(theta_1)

    # pd1 may hide pd2
    draw_rect(im, pd2_color, pd1_end_x, pd1_end_y, w2, h2, theta_2)
    draw_rect(im, pd1_color, center[0], center[1], w1, h1, theta_1)
    
    im = im.resize((128, 128))
    return im
    
def gen_data(data_filepath, num_seq, num_frm, seed=0):
    rng = np.random.default_rng(seed)
    states = np.zeros((num_seq, num_frm, 4))

    for n in tqdm(range(num_seq)):
        seq_filepath = os.path.join(data_filepath, str(n))
        mkdir(seq_filepath)
        states[n, :, :] = engine(rng, num_frm)
        for k in range(num_frm):
            im = render(states[n, k, 0], states[n, k, 2])
            im.save(os.path.join(seq_filepath, str(k)+'.png'))
    
    np.save('./data/physics_prediction_v2/data/double_pendulum/states.npy', states)

if __name__ == '__main__':
    data_filepath = './data/physics_prediction_v2/data/double_pendulum'
    gen_data(data_filepath, num_seq=1200, num_frm=60)