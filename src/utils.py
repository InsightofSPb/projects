from os import path
from pandas import read_csv, DataFrame

from paths import ANNOTATION_PATH

def load_loss():
    pass

def read_dataset_data() -> DataFrame:
    df = read_csv(path.join(ANNOTATION_PATH, 'annotations,csv'), sep='\t')
    return df