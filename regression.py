import os
import sys
import csv
import yaml
import sympy
import pprint
import shutil
import numpy as np
from utils.helpers import *
from munch import munchify
from pysr import PySRRegressor

def custom_loss_string(id):
    return f"""
    function my_loss(tree, dataset::Dataset{{T,L}}, options, idx) where {{T,L}}
        prediction, flag = eval_tree_array(tree, dataset.X, options)
        if !flag
            return L(Inf)
        end
        y = dataset.y
        loss = L(sum(i -> abs2(prediction[i] - y[i]), eachindex(y)) / length(y))

        # Check if any node in the tree is feature i
        # (note that Julia indexes from 1 rather than 0)
        for i = 1:{id}
            contains_xi = any(
                node -> (
                    node.degree == 0
                    && !(node.constant)
                    && node.feature == i
                ),
                tree
            )
            if !(contains_xi)
                # penalty term
                loss += L(1000)
            end
        end
        return loss
    end
    """
    
def load_config(filepath):
    with open(filepath, 'r') as stream:
        try:
            trainer_params = yaml.safe_load(stream)
            return trainer_params
        except yaml.YAMLError as exc:
            print(exc)

def cleanup(cfg, sr_dir, full_simplify=True):

    data = [['Variable', 'Loss', 'Equation']]
    for k in range(cfg.intrinsic_dimension):
        fit_filepath = os.path.join(sr_dir, f'model.csv.out{k+1}')
        fit = np.genfromtxt(fit_filepath, delimiter=',', dtype=None, encoding='utf-8')
        loss, eq = fit[1:, 1].astype(float), fit[1:, 2]
        idx = np.argmin(loss)
        data.append([k+1, loss[idx], simplify(eq[idx], 1e-6, full_simplify)])
        
    with open(os.path.join(sr_dir, 'fit.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(data)

    for k in range(cfg.intrinsic_dimension):
        fit_filepath = os.path.join(sr_dir, f'model.csv.out{k+1}')
        os.remove(f'{fit_filepath}.bkup')
        os.remove(fit_filepath)

def main():
    config_filepath = str(sys.argv[1])
    cfg = load_config(filepath=config_filepath)
    pprint.pprint(cfg)
    cfg = munchify(cfg)

    log_dir_name = '_'.join([cfg.dataset, cfg.model_name.replace('sr', '64'), str(cfg.seed)])
    log_dir = os.path.join(cfg.log_dir, log_dir_name)
    variable_dir = os.path.join(log_dir, 'variables_train')
    ids = np.load(os.path.join(variable_dir, 'ids.npy'))
    ids = [int(id[:id.index('_')]) for id in ids][::cfg.num_frames-1]

    states_dir = os.path.join(cfg.data_filepath, 'collect', cfg.dataset, 'states.npy')
    states_raw = np.load(states_dir)
    states = states_raw[ids, :cfg.num_frames-1]
    states = np.reshape(states, (-1, cfg.intrinsic_dimension))
    latent = np.load(os.path.join(variable_dir, 'refine_mu.npy'))
    objective = custom_loss_string(cfg.intrinsic_dimension)
    for k,v in cfg.constraints.items():
        if isinstance(v, list):
            cfg.constraints[k] = tuple(v)

    sr_dir = os.path.join(log_dir, 'symbolic_regression')
    if os.path.exists(sr_dir):
        shutil.rmtree(sr_dir)
    os.makedirs(sr_dir)

    model = PySRRegressor(
        niterations=cfg.niterations,
        maxsize=cfg.maxsize,
        warmup_maxsize_by=cfg.warmup_maxsize_by,
        loss_function=objective,
        batching=True,
        batch_size=cfg.batch_size,
        model_selection='accuracy',
        complexity_of_constants=cfg.complexity_of_constants,
        complexity_of_variables=cfg.complexity_of_variables,
        binary_operators=cfg.binary_operators,
        unary_operators=cfg.unary_operators,
        constraints=cfg.constraints,
        nested_constraints=cfg.nested_constraints,
        equation_file=os.path.join(sr_dir, 'model.csv'),
        procs=8,
        # for deterministic behavior
        # random_state=cfg.seed,
        # deterministic=True,
        # procs=0,
    )

    model.fit(states, latent)
    cleanup(cfg, sr_dir)
    
if __name__ == '__main__':
    main()