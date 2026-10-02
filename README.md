# The machine learning based pipeline (TxCyto) to estimate cytokine activity in a tissue sample.
## Overview
TxCyto is a deep-learning based framework to estimate cytokine activity from whole transcriptome of a sample.
Genes function through an interconnected network of different regulatory molecules such as cytokine, transcription factors, or miRNA. Any perturbation in the regulatory molecules propagate through this network and have a predictable impact on the global transcriptome of the tissue.  Based on this premise, we developed TxCyto - a deep learning-based framework that infers the activity of cytokines, directly from the whole transcriptome profile of a sample. 
## Workflow
The TxCyot is based on fully connected neural network with three layers. An input layer consisting of transcriptomics feature floowed by three full connected headen layer.
Input → Dense(128, ReLU) → Dense(64, ReLU) → Dense(32, ReLU) → Output(1)
Models are trained using Adams optimizer with mean squared error (MSE) as the loss function. The output represent the inferred acivtivy of the corresponding cytokine.
## Key dependencies
    Python 3.9.15
    Numpy 1.24.4
    Pandas 2.0.3
    Scikit-learn 1.3.0
    Tensorflow 2.7.0
    Keras. 2.7.0
    Scipy.  1.11.1
    Matplotlib. 3.7.2
## Repository structure
