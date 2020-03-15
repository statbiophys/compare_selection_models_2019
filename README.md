## On generative models of T cell receptor sequences

Written by Giulio Isacchini, MPIDS Göttingen - ENS Paris.

Last updated on 15-03-2020

Reference: On generative models of T cell receptor sequences, Giulio Isacchini, Zachary Sethna, Yuval Elhanati, Armita Nourmohammad, Aleksandra M. Walczak, Thierry Mora, https://arxiv.org/abs/1911.12279

=== Reproduce Plots ===

In order to reproduce the plots you need to unzip the files input.zip and vampire-emerson.zip.

1) Sample data.

To sample the data for all figures, run sample_data.py

2) Infer the models.

To infer the model for fig1, run fig1_infer.py

To infer the model for fig2, run fig2_infer.py

3) Plot the results.

To recreate Fig 1, run fig1_plot.py

To recreate Fig 2a run fig2a_plot.py

To recreate Fig 2b, run fig2b_evaluate.py (in the vampire environment) and then fig2b_plot.py

=== Requisites ===

- olga
- tensorflow
- numpy
- pandas
- scipy
- matplotlib

This directory includes a minimal version of the Sonia package. The full package is available in https://github.com/statbiophys/SONIA/ 