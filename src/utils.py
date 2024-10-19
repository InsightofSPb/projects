import importlib
from os import path
from pandas import read_csv, DataFrame

from src.paths import ANNOTATION_PATH

def load_loss(obj_path: str, def_obj_path: str = ''):
    obj_path_list = obj_path.rsplit('.', 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else def_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f'Object `{obj_name}` cannot be loaded from `{obj_path}`.')
    return getattr(module_obj, obj_name)

def read_dataset_data() -> DataFrame:
    df = read_csv(path.join(ANNOTATION_PATH, 'annotations.csv'), sep='\t')
    return df