import numpy as np
import pandas as pd
import subprocess
import olga.load_model as load_model
import olga.sequence_generation as seq_gen
import os
import random as rn

def run_terminal(string):
    return subprocess.Popen(string, shell=True, stdout=subprocess.PIPE,stderr = subprocess.PIPE).communicate()

#set seed for reproducibility
np.random.seed(42)
rn.seed(1234)

##############################
#####sample deneuter data#####
##############################

print 'sample deneuter'

#you need to process the data as it is explained in vampire-analysis-1 and take care to keep the exact train test split that they use.
deneuter_directory='input/out_deneuter/'
data=pd.read_csv(deneuter_directory+'deneuter.train.csv',sep=',').sample(n=int(15e4)).reset_index(drop=True)
data.drop_duplicates(['amino_acid','v_gene','j_gene']).reset_index(drop=True).iloc[:int(1e5)].to_csv('sampled_data/deneuter_data.csv',index=False)


###########################
###sample seqs from olga###
###########################

print 'sample olga'
olga_model='universal_model'

#Load generative model for emerson
pathdir= os.getcwd()
main_folder = os.path.join(pathdir,olga_model)
params_file_name = os.path.join(main_folder,'model_params.txt')
marginals_file_name = os.path.join(main_folder,'model_marginals.txt')
V_anchor_pos_file = os.path.join(main_folder,'V_gene_CDR3_anchors.csv')
J_anchor_pos_file = os.path.join(main_folder,'J_gene_CDR3_anchors.csv')

genomic_data = load_model.GenomicDataVDJ()
genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)

generative_model = load_model.GenerativeModelVDJ()
generative_model.load_and_process_igor_model(marginals_file_name)        

seq_gen_model = seq_gen.SequenceGenerationVDJ(generative_model, genomic_data)

seqs_generated=[seq_gen_model.gen_rnd_prod_CDR3() for i in range(int(3e6))]
seqs = [[seq[1], genomic_data.genV[seq[2]][0].split('*')[0], genomic_data.genJ[seq[3]][0].split('*')[0]] for seq in seqs_generated]
df=pd.DataFrame(seqs,columns=['amino_acid','v_gene','j_gene'])
df.to_csv('sampled_data/generated_sequences.csv',index=False)

olga_model='deneuter_model'
print 'sample deneuter'

#Load generative model for emerson
pathdir= os.getcwd()
main_folder = os.path.join(pathdir,olga_model)
params_file_name = os.path.join(main_folder,'model_params.txt')
marginals_file_name = os.path.join(main_folder,'model_marginals.txt')
V_anchor_pos_file = os.path.join(main_folder,'V_gene_CDR3_anchors.csv')
J_anchor_pos_file = os.path.join(main_folder,'J_gene_CDR3_anchors.csv')

genomic_data = load_model.GenomicDataVDJ()
genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)

generative_model = load_model.GenerativeModelVDJ()
generative_model.load_and_process_igor_model(marginals_file_name)        

seq_gen_model = seq_gen.SequenceGenerationVDJ(generative_model, genomic_data)

seqs_generated=[seq_gen_model.gen_rnd_prod_CDR3() for i in range(int(3e6))]
seqs = [[seq[1], genomic_data.genV[seq[2]][0].split('*')[0], genomic_data.genJ[seq[3]][0].split('*')[0]] for seq in seqs_generated]
df=pd.DataFrame(seqs,columns=['amino_acid','v_gene','j_gene'])
df.to_csv('sampled_data/generated_sequences_deneuter.csv',index=False)
