import datetime
import os, logging
import numpy as np
import pandas as pd
import configparser
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def get_filename(PATH, head, tail):
    i = 0
    today = datetime.datetime.now()
    today = today.strftime('%Y%m%d')
    if not os.path.exists(os.path.join(PATH, today)):
        os.mkdir(os.path.join(PATH, today))
    name = today+'/'+head+'-'+today+'-'+'%02d'%i+tail
    while os.path.exists(os.path.join(PATH, name)):
        i += 1
        name = today+'/'+head+'-'+today+'-'+'%02d'%i+tail
    return name
    
def get_logger():
    return