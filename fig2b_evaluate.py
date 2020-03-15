import numpy as np 
import pandas as pd
import timeit
import subprocess
def run_terminal(string):
    return subprocess.Popen(string, shell=True, stdout=subprocess.PIPE,stderr = subprocess.PIPE).communicate()

start = timeit.default_timer()
data2='sampled_data/generated_sonia_leftright_adaptive.csv sampled_data/generated_sonia_leftright_adaptive.pvae.csv'
params='input/_output_deneuter-2019-02-07/deneuter-2019-02-07.train/0.75/basic/model_params.json input/_output_deneuter-2019-02-07/deneuter-2019-02-07.train/0.75/basic/best_weights.h5 '
x=run_terminal('tcr-vae pvae '+params+data2)
print(x[1])
stop = timeit.default_timer()
print ('Time', stop- start)