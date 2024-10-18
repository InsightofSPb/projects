from os import path

PROJECT_PATH = path.abspath(path.join(path.dirname(path.realpath(__file__)), '../'))
CONFIG_PATH = path.join(PROJECT_PATH, 'configs')
IMAGES_PATH = path.join(PROJECT_PATH, 'data/images')
ANNOTATION_PATH = path.join(PROJECT_PATH, 'data')
EXPERIMENTS_PATH = path.join(PROJECT_PATH, 'experiments')