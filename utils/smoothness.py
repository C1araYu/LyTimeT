import os
import sys
import yaml
import numpy as np
from munch import munchify

def load_config(filepath):
    with open(filepath, 'r') as stream:
        try:
            trainer_params = yaml.safe_load(stream)
            return trainer_params
        except yaml.YAMLError as exc:
            print(exc)
            
def compute_smoothness(dataset):
    configs_dir = sys.argv[2]
    loss = []
    for config_filepath in os.listdir(configs_dir):
        cfg = load_config(filepath=os.path.join(configs_dir, config_filepath))
        cfg = munchify(cfg)
        log_dir_name = '_'.join([cfg.dataset, cfg.model_name, str(cfg.seed)])
        log_dir = os.path.join(cfg.log_dir, log_dir_name)
        vars_filepath = os.path.join(log_dir, 'variables')
        vars = np.load(os.path.join(vars_filepath, 'refine_latent.npy'))
        ids = np.load(os.path.join(vars_filepath, 'ids.npy'))
        
        smoothness = 0
        v_idxs = split_by_video(ids)
        for _, v_idx in v_idxs.items():
            latent = vars[v_idx]
            smoothness += smoothness_loss(latent)
        smoothness /= len(v_idxs)
        loss.append(smoothness)
    return loss

def split_by_video(ids):
    v_idxs = {}
    for k, id in enumerate(ids):
        vnum = id.split('_')[0]
        if vnum not in v_idxs.keys():
            v_idxs[vnum] = []
        v_idxs[vnum].append(k)
    return v_idxs

def smoothness_loss(latent):
    loss = 0
    derivs = 4
    maxs = np.amax(latent, axis=0, keepdims=True)
    mins = np.amin(latent, axis=0, keepdims=True)
    scaled_latent = latent / (maxs - mins + 1e-12)
    last_deriv = scaled_latent
    for k in range(derivs):
        deriv = last_deriv[1:] - last_deriv[:-1]
        deriv_loss = np.mean(np.abs(deriv))   # sum along time dimension
        loss += (5.0**(k+1)) * deriv_loss
        last_deriv = deriv
    return loss

def main(dataset):
    loss = compute_smoothness(dataset)
    mean = np.round(np.mean(loss), 2)
    std = np.round(np.std(loss), 2)
    print(f'Smoothness: {mean} ± {std}')
    

if __name__ == '__main__':
    dataset = sys.argv[1]
    main(dataset)