import os
import sys
import yaml
import numpy as np
from munch import munchify
from scipy.stats import gaussian_kde

def load_config(filepath):
    with open(filepath, 'r') as stream:
        try:
            trainer_params = yaml.safe_load(stream)
            return trainer_params
        except yaml.YAMLError as exc:
            print(exc)

def main(dataset, samples=10**4):
    configs_dir = sys.argv[2]
    hum_vars, ai_vars, mi = [], [], []
    for config_filepath in os.listdir(configs_dir):
        cfg = load_config(filepath=os.path.join(configs_dir, config_filepath))
        cfg = munchify(cfg)
        log_dir_name = '_'.join([cfg.dataset, cfg.model_name, str(cfg.seed)])
        log_dir = os.path.join(cfg.log_dir, log_dir_name)
        vars_filepath = os.path.join(log_dir, 'variables')
        vars = np.load(os.path.join(vars_filepath, 'refine_latent.npy'))
        ids = np.load(os.path.join(vars_filepath, 'ids.npy'))
        ids = [int(i[:i.index('_')]) for i in ids][::59]
        ai_vars.append(vars)
        
        hum_vars_filepath = os.path.join(cfg.data_filepath, cfg.dataset, 'states.npy')
        vars = np.load(hum_vars_filepath)[ids, :59]
        vars = np.reshape(vars, (-1, vars.shape[-1]))
        hum_vars.append(vars)

    ID = hum_vars[0].shape[-1]
    for seed in range(1,4):
        x = hum_vars[seed-1].transpose()
        y = ai_vars[seed-1].transpose()
        xy = np.concatenate([x,y], axis=0)
        Px = gaussian_kde(x)
        Py = gaussian_kde(y)
        Pxy = gaussian_kde(xy)
        
        xy = Pxy.resample(size=samples, seed=42)
        x, y = xy[:ID], xy[ID:]
        xy = np.concatenate([x,y], axis=0)
        px = Px.logpdf(x)
        py = Py.logpdf(y)
        pxy = Pxy.logpdf(xy)
        f = pxy - px - py
        mi.append(f.mean())
    mi = np.array(mi)
    mean = np.round(np.mean(mi), 2)
    std = np.round(np.std(mi), 2)
    print(f'Mutual Info: {mean} ± {std}')

if __name__ == '__main__':
    dataset = sys.argv[1]
    main(dataset)