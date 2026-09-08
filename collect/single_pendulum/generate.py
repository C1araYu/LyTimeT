import os
import sys
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
from scipy.integrate import solve_ivp

def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def engine(rng, num_frm, fps=60):
    m = 1.0
    l = 0.5
    g = 9.81
    dt = 1.0 / fps
    t_eval = np.arange(num_frm) * dt

    # y = [theta, omega]
    f = lambda t, y: [y[1], -3*g/(2*l) * np.sin(y[0])]
    initial_state = [rng.uniform(-np.pi, np.pi), rng.uniform(-10, 10)]
    sol = solve_ivp(f, [t_eval[0], t_eval[-1]], initial_state, t_eval=t_eval, rtol=1e-6)
    
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
    
def render(theta):
    bg_color = (215, 205, 192)
    pd_color = (17, 93, 234)
    img_size = (1600, 1600)

    im = Image.new('RGB', img_size, bg_color)
    center = (800, 800)
    
    w, h = 90, 500
    pd1_end_x = center[0] + (h-35) * np.sin(theta)
    pd1_end_y = center[1] + (h-35) * np.cos(theta)
    
    draw_rect(im, pd_color, center[0], center[1], w, h, theta)
    
    im = im.resize((128, 128))
    return im

def gen_data(data_filepath, num_seq, num_frm, seed=0):
    rng = np.random.default_rng(seed)
    states = np.zeros((num_seq, num_frm, 2))
    
    for n in tqdm(range(num_seq)):
        seq_filepath = os.path.join(data_filepath, str(n))
        mkdir(seq_filepath)
        states[n, :, :] = engine(rng, num_frm)
        for k in range(num_frm):
            im = render(states[n, k, 0])
            im.save(os.path.join(seq_filepath, str(k)+'.png'))
    
    np.save('/root/data/physics_prediction_v2/data/single_pendulum/states.npy', states)

if __name__ == '__main__':
    data_filepath = '/root/data/physics_prediction_v2/data/single_pendulum'
    gen_data(data_filepath, num_seq=1200, num_frm=60)