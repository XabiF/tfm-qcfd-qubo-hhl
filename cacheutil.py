import numpy as np
import os
import shutil

def get_save_dir(basedir):
    # Get directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the full path
    save_dir = os.path.join(script_dir, basedir)
    os.makedirs(save_dir, exist_ok=True)  # create if it doesn’t exist
    
    return save_dir

def get_save_path(basedir, filename):
    return os.path.join(get_save_dir(basedir), filename)

def purge_save_dir(basedir):
    save_dir = get_save_dir(basedir)
    shutil.rmtree(save_dir, ignore_errors=True)

def save_data(basedir, filename, data):
    np.savetxt(get_save_path(basedir, filename), data[:, None], delimiter=",")

def try_load_data(basedir, filename):
    path = get_save_path(basedir, filename)
    if os.path.isfile(path):
        return np.loadtxt(path, delimiter=",")
    else:
        return None
