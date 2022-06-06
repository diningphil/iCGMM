# The Infinite Contextual Graph Markov Model (iCGMM)

## Summary
iCGMM is a deep unsupervised model for graph learning. It extends the [Contextual Graph Markov Model](https://github.com/diningphil/CGMM) (CGMM) so that each layer's complexity changes depending on the difficulty of the learning task. 
In addition, iCGMM automatizes the choice of almost all its hyper-parameters, which are estimated during inference (Gibbs sampling).

## This repo
The library includes data and scripts to reproduce the graph classification experiments reported in the paper describing the method.

This research software is provided as is. If you happen to use or modify this code, please remember to cite the foundation papers:

*Castellana Daniele, Errica Federico, Bacciu Davide, Micheli Alessio: The Infinite Contextual Graph Markov Model. Proceedings of the 39th International Conference on Machine Learning, 2022.*

### Usage

This repo builds upon [PyDGN](https://github.com/diningphil/PyDGN), a framework to easily develop and test new DGNs.
See how to construct your dataset and then train your model there.

This repo assumes PyDGN 1.0.9 is used. Compatibility with future versions is not guaranteed.

The evaluation is carried out in two steps:
- Generate the unsupervised graph embeddings
- Apply a classifier on top

We designed two separate experiments to avoid recomputing the embeddings each time. First, use the `iCGMM_CONFIGS/*embedding*.yml` config file to create the embeddings,
specifying the hyper-parameter ranges as in the paper. Then, use the `iCGMM_CONFIGS/*classifier*.yml` config file to launch
the classification experiments. It is mandatory that the configuration for the classifier holds exactly the same hyper-parameter grid
for the embedding part, so that the embeddings can be correctly loaded.

## Launch Exp:

#### Build dataset and data splits (follow PyDGN tutorial)
You can use the data splits we provided for graph classification tasks, taken from our [ICLR 2020](https://arxiv.org/abs/1912.09893) paper on reproducibility.

For instance:

    pydgn-dataset --config-file DATA_CONFIGS/config_PROTEINS.yml

#### Train the model

    pydgn-train  --config-file iCGMM_CONFIGS/config_iCGMM_seq_embedding_CHEMICAL.yml
    pydgn-train  --config-file iCGMM_CONFIGS/config_iCGMM_seq_classifier_CHEMICAL.yml

#### A few things to notice

- The first script will store node embeddings in a folder specified in the configuration files. These can take space, so ensure you monitor disk usage
- Be aware of the hardware and parallelism required in the configuration files. Refer to PyDGN if you don't know how to modify them.
