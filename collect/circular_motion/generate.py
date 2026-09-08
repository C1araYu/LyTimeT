import os
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw

def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def engine(rng, num_frm, fps=60):
    dt = 1.0 / fps
    initial_state = [
        rng.uniform(0, 2*np.pi),
        rng.choice((-1, 1)) * rng.uniform(2*np.pi, 6*np.pi),
    ]
    states = np.zeros((num_frm, 2))
    states[0] = initial_state
    for k in range(1, num_frm):
        states[k, 0] = states[k-1, 0] + dt * states[k-1, 1]
        states[k, 1] = states[k-1, 1]
    return states

def render(theta):
    bg_color = (215, 205, 192)
    obj_color = (17, 93, 234)
    img_size = (1600, 1600)

    im = Image.new('RGB', img_size, bg_color)
    draw = ImageDraw.Draw(im)
    center = (800, 800)
    
    radius = 400
    obj_size = 100
    x = center[0] + radius * np.cos(theta)
    y = center[1] - radius * np.sin(theta)
    pos = (x - obj_size, y - obj_size, x + obj_size, y + obj_size)
    draw.ellipse(pos, fill=obj_color)
    
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
            
    np.save('./data/physics_prediction_v2/data/circular_motion/states.npy', states)

if __name__ == '__main__':
    data_filepath = './data/physics_prediction_v2/data/circular_motion'
    gen_data(data_filepath, num_seq=1200, num_frm=60)