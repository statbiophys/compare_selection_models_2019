import numpy as np 
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from SONIA.evaluate_model import EvaluateModel
from SONIA.sequence_generation import SequenceGeneration
from SONIA.sonia_length_pos import SoniaLengthPos
from SONIA.sonia_vjl import SoniaVJL
from SONIA.sonia_leftpos_rightpos import SoniaLeftposRightpos
import olga.load_model as olga_load_model
import olga.generation_probability as generation_probability
import olga.sequence_generation as seq_gen
import os
import process_adaptive as pa
from scipy import stats
import matplotlib.pyplot as plt
import re
import keras.backend as K
import tensorflow as tf
import random as rn

# set seeds for reproducibility
np.random.seed(42)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
    
#load data
data=pd.read_csv('sampled_data/deneuter_data.csv')
data=list(pa.adaptive2olga(data).values)
gen=list(pd.read_csv('sampled_data/generated_sequences.csv').values.astype(np.str)[:int(15e5)])
print len(data),len(gen)
# gen model
main_folder='deneuter_model'
params_file_name = os.path.join(main_folder,'model_params.txt')
marginals_file_name = os.path.join(main_folder,'model_marginals.txt')
V_anchor_pos_file = os.path.join(main_folder,'V_gene_CDR3_anchors.csv')
J_anchor_pos_file = os.path.join(main_folder,'J_gene_CDR3_anchors.csv')

genomic_data = olga_load_model.GenomicDataVDJ()
genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)
generative_model = olga_load_model.GenerativeModelVDJ()
generative_model.load_and_process_igor_model(marginals_file_name)
pgen_model = seq_gen.SequenceGenerationVDJ(generative_model, genomic_data)

#LENGTHPOS MODEL
qm = SoniaLengthPos(data_seqs = data, gen_seqs=gen,custom_pgen_model='deneuter_model')
qm.infer_selection(epochs=50,batch_size=int(1e4),validation_split=0.01)
qm.save_model('selection_models/deneuter_model_lengthpos')
ev=SequenceGeneration(sonia_model=qm,custom_olga_model=pgen_model,custom_genomic_data=genomic_data)
seqs_sonia=ev.generate_sequences_post(int(2e4))
pd.DataFrame(seqs_sonia,columns=['amino_acid','v_gene','j_gene']).to_csv('sampled_data/generated_sonia_lengthpos.csv',index=False)

#LEFTRIGHT MODEL
qm = SoniaLeftposRightpos(data_seqs = data, gen_seqs=gen,custom_pgen_model='deneuter_model')
qm.infer_selection(epochs=50,batch_size=int(1e4),validation_split=0.01)
qm.save_model('selection_models/deneuter_model_leftright')
ev=SequenceGeneration(sonia_model=qm,custom_olga_model=pgen_model,custom_genomic_data=genomic_data)
seqs_sonia=ev.generate_sequences_post(int(2e4))
pd.DataFrame(seqs_sonia,columns=['amino_acid','v_gene','j_gene']).to_csv('sampled_data/generated_sonia_leftright.csv',index=False)

# OLGA Q MODEL
qm = SoniaVJL(data_seqs = data, gen_seqs=gen,custom_pgen_model='deneuter_model')
qm.infer_selection(epochs=50,batch_size=int(1e4))
qm.save_model('selection_models/deneuter_model_VJL')
ev=SequenceGeneration(sonia_model=qm,custom_olga_model=pgen_model,custom_genomic_data=genomic_data)
seqs_sonia=ev.generate_sequences_post(int(2e4))
pd.DataFrame(seqs_sonia,columns=['amino_acid','v_gene','j_gene']).to_csv('sampled_data/generated_sonia_VJL.csv',index=False)

# convert to adaptive format for evaluation
pa.olga2adaptive(pd.read_csv('sampled_data/generated_sonia_lengthpos.csv')).to_csv('sampled_data/generated_sonia_lengthpos_adaptive.csv')
pa.olga2adaptive(pd.read_csv('sampled_data/generated_sonia_leftright.csv')).to_csv('sampled_data/generated_sonia_leftright_adaptive.csv')
pa.olga2adaptive(pd.read_csv('sampled_data/generated_sonia_VJL.csv')).to_csv('sampled_data/generated_sonia_VJL_adaptive.csv')