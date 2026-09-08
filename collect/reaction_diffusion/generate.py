import os
import shutil
from tqdm import tqdm
import numpy as np
import scipy.io as sio
from PIL import Image
from matplotlib import cm

def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def gen_data(data_filepath, num_frm):
    data = sio.loadmat('/root/tide/collect/reaction_diffusion/reaction_diffusion.mat')
    num_seq = int(data['t'].size / num_frm)
    u_min, u_max = -1, 1

    for n in tqdm(range(num_seq)):
        seq_filepath = os.path.join(data_filepath, str(n))
        mkdir(seq_filepath)
        for k in range(num_frm):
            u = data['uf'][:, :, n*num_frm+k]
            u = (u - u_min) / (u_max - u_min)
            u = cm.viridis(u)[:,:,:3]
            u = np.uint8(u*255)
            im = Image.fromarray(u)
            im.save(os.path.join(seq_filepath, str(k)+'.png'))

if __name__ == '__main__':
    data_filepath = '/root/data/physics_prediction_v2/data/reaction_diffusion'
    gen_data(data_filepath, num_frm=100)