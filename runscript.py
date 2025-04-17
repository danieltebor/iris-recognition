import argparse
import importlib.util
import os
import yaml

from iris.common.constants import *


def main():
    argparser = argparse.ArgumentParser('Run a script with a configuration file')
    argparser.add_argument(
        '--script', type=str, required=False,
        help='Name of the script to run'
    )
    argparser.add_argument(
        '--mode', type=str, required=False,
        help='Mode to run the script in (train, eval, train-and-eval)'
    )
    argparser.add_argument(
        '--config', type=str, required=False,
        help='Name of the configuration file'
    )
    args = argparser.parse_args()
    
    mode = _get_mode(args)
    script = _get_script(args)
    script_hyphens = script.replace('_', '-')
    script = script + '.py'
    config = _get_config(args, script_hyphens)

    print('\nSelected options:')
    print(f'Mode: {mode}')
    print(f'Script: {script}')
    print(f'Config: {config}')
    
    print('\nBuilding script')
    spec = importlib.util.spec_from_file_location(script, os.path.join(SCRIPT_DIR, script))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    with open(os.path.join(CONFIG_DIR, script_hyphens, config), 'r') as f:
        config = yaml.safe_load(f)
        
    print('\nSETTINGS:')
    for key, value in config.items():
        print(f'{key}: {value}')
    
    mode = mode.split('-')
    model_dir = None
    if 'train' in mode:
        if hasattr(module, 'train_model'):
            model_dir = module.train_model(config)
        else:
            raise ValueError('train_model not found in script')
        
    if 'eval' in mode:
        if model_dir is None:
            model_dir = _get_model_dir(script_hyphens, config)
        
        if hasattr(module, 'eval_model'):
            module.eval_model(config, model_dir)
        else:
            raise ValueError('eval_model not found in script')

def _get_mode(args) -> str:
    valid_modes = ['train', 'eval', 'train-and-eval']
    
    if args.mode:
        if args.mode not in valid_modes:
            raise ValueError(f'Invalid mode. Must be one of: {valid_modes}')
        return args.mode
    
    print('Defaulting to train-and-eval mode')
    return 'train-and-eval'

def _get_script(args) -> str:
    available_scripts = os.listdir(SCRIPT_DIR)
    
    if args.script:
        scripts_no_ext = [script.split('.')[0] for script in available_scripts]
        if args.script not in available_scripts + scripts_no_ext:
            raise ValueError('Invalid script name')
        return args.script.split('.')[0]
    
    print('Scripts:')
    for i, script in enumerate(available_scripts):
        print(f'{i+1}. {script}')
    script_idx = _get_numerical_input('Select a script: ', 1, len(available_scripts)) - 1
    return available_scripts[script_idx].split('.')[0]

def _get_config(args, script: str) -> str:
    config_path = os.path.join(CONFIG_DIR, script)
    available_configs = os.listdir(config_path)
    
    if args.config:
        configs_no_ext = [config.split('.')[0] for config in available_configs]
        if args.config not in available_configs + configs_no_ext:
            raise ValueError('Invalid config name')
        return args.config.split('.')[0] + '.yaml'
    
    print('Configs:')
    for i, config in enumerate(available_configs):
        print(f'{i+1}. {config}')
    config_idx = _get_numerical_input('Select a config: ', 1, len(available_configs)) - 1
    return available_configs[config_idx].split('.')[0] + '.yaml'

def _get_model_dir(script: str, config: str) -> str:
    model_base_dir = os.path.join(OUT_DIR, script, config['experiment_name'])
    print('\nModels:')
    models = os.listdir(model_base_dir)
    for i, model in enumerate(models):
        print(f'{i+1}. {model}')
    model_idx = _get_numerical_input('Select a model (Enter for latest): ', 1, len(models), True)
    if model_idx is None:
        return os.path.join(model_base_dir, models[-1])
    return os.path.join(model_base_dir, models[model_idx - 1])

def _get_numerical_input(prompt: str, lower_bound: int, upper_bound: int, allow_empty: bool = False) -> int:
    while True:
        try:
            num = input(prompt)
            if num == '':
                if allow_empty:
                    return None
                raise ValueError('Input cannot be empty')
            if not num.isdigit():
                raise ValueError('Input must be a number')
            num = int(num)
            if num < lower_bound or num > upper_bound:
                raise ValueError(f'Number must be between {lower_bound} and {upper_bound}')
            return num
        except ValueError as e:
            print(e)

if __name__ == '__main__':
    main()