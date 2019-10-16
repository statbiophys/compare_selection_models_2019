## Comparison of models of TCR-beta repertoires

Written by Giulio Isacchini, MPIDS Göttingen - ENS Paris.

Last updated on 13-10-2019

Reference: in preparation.

=== Reproduce Plots ===

In order to reproduce the plots you need to unzip the file sampled_data.zip. You need to additionally download the data from https://zenodo.org/record/2619576#.XKElTrfYphE, copy the folder in the main directory and rename it as 'input'.

To recreate Fig 1a, run fig1a_plot.py
To recreate Fig 1b, run fig1b_plot.py
To recreate Fig 1c, run fig1c_plot.py

You can also reinfer the models:

To infer the model for fig1a, run fig1a_infer.py
To infer the model for fig1b and fig1c, run fig1bc_infer.py

=== Train Data ===

The training data is included in the sampled_data directory. However if you want to reproduce the training data too, a couple of more steps are needed:

First you need to additionally download the following data:
- https://clients.adaptivebiotech.com/pub/emerson-2017-natgen 
- https://clients.adaptivebiotech.com/pub/deneuter-2018-cmvserostatus 

Process the data files from emerson-2017-natgen using the preprocess_adaptive.py script of the vampire package and add them to a folder called emerson_processed in the main directory.
Process the data from deneuter-2018-cmvserostatus with the train_test.sh script of the vampire package. It should create a file in the input/out_deneuter folder.
We recommmend particular care in the definition of the same train-test split for the processing of deneuter-2018-cmvserostatus data as the one of the orginal paper.
More precise explanation can be found in https://github.com/matsengrp/vampire-analysis-1/.

To re-sample the training data, run sample_data.py

=== Estimate Pvae ===

Pvae for sequences sampled from the sonia model has been already estimated and included in the sampled_data folder.

However, if you want to estimate it yourself you need to:
-install the vampire package
-convert the file generated_sonia.csv (output of fig1bc_infer.py script) to a format compatible with adaptive using the olga2adaptive function of the vampire package
-run the command 'tcr-vae pvae' on the new data and specify the vae model.


=== Requisites ===

- olga
- tensorflow
- keras
- numpy
- pandas
- scipy
- matplotlib

This directory includes a minimal version of the Sonia package. The full package will is available in https://github.com/statbiophys/SONIA/ 