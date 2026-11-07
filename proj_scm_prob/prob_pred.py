import os
import pickle

def parse_model_cp_name(model_cp_name: str):
    parts = model_cp_name.split('-')
    if len(parts) < 3 or parts[0] != 'model':
        raise ValueError(
            'model_cp_name must follow format model-<model_name_full>-<fold_id>, '
            f'got: {model_cp_name}'
        )

    model_type = parts[1]
    model_name_full = '-'.join(parts[1:-1])
    fold_id_i = parts[-1]
    return model_type, model_name_full, fold_id_i

def load_model_from_checkpoint(
    model_name_full: str,
    fold_id_i: str,
    dir_save_full: str,
):
    dir_save_model = os.path.join(dir_save_full, f'model-{model_name_full}-{fold_id_i}.pkl')
    with open(dir_save_model, 'rb') as f:
        model_out = pickle.load(f)
    return model_out