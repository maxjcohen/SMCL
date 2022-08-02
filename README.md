# Sequential Monte Carlo Layer
This repo contains the SMCL pytorch library as well as full model implementation examples.

## SMC Layer (SMCL) library
This package only contains the SMCL module, which inherits from the torch `RNN` class. It can be used by itself, but because of its high computation cost when dimensionality increases, it is advised to plug it on top of an existing model. Note that when combined this way, it *has* to be plugged on *top*: in other words, it should be the last layer of the model.

When only inputs (referred to as "commands" in this repo) are provided, the SMC layer behave like a traditional RNN. When additional observations are made available, the model performs SMC computations (generate particles, compute importance weights, re sample, etc.). Note that it is possible to trigger the SMC behavior without transmitting any or part of the observations by replacing values with `NaN`.

## Usage examples
Example are currently derived for each dataset, we will consider the `energy` dataset for this explanation. Is the following example, we define a simple model combining an input model (a three-layered GRU) with a SMC layer (see `src/modules.py`). Examples are divided in three steps:

 - **Pretrain:** we train the entire model in a classic fashion (deterministic forward pass, traditional MSE loss computation and reduction through gradient descent). The corresponding script is located at [`scripts/energy/pretrain.py`](scripts/energy/pretrain.py).
 - **Finetune:** the SMC last layer weights are finetuned. As the loss function is not informative (it's only used for its gradient), we display the previous MSE loss during validation). The corresponding script is located at  [`scripts/energy/smcm.py`](scripts/energy/smcm.py).
 - **Visualize:** dataset, predictions, smoothing and filter among others are displayed in the `visualization_energy.ipynb` notebook.
