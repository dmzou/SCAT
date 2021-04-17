# SCAT: Scattering Transform on Graphs

This repository maintains codes used for two papers [1, 2]. The scattering transform on graphs uses multi-layer and multi-scale structure based on graph wavelets and achieves robust representation of input graph signals. We specifially illustrate a graph generation model based on the graph scattering transform, which is naturally composed of an encoder and a decoder. The encoder is a Gaussianized graph scattering transform, which is robust to signal and graph manipulation. The decoder is a simple fully connected network that is adapted to specific tasks, such as link prediction, signal generation on graphs and full graph and signal generation.

## Getting Started

This repository contains the graph scattering transform. To use it, see [scat.py](scat.py) and the comments in that file. See [example.py](example.py) for the illustrative example of Cora in [1]. This repository also specifically demonstates the implementation of the experiments described in [2]. The codes can be used for implementing the following models:
- Link prediction (SCAT-S, SCAT-D)
```
python trainCora.py -s S [S/D] -d cora [cora/citeseer/pubmed]
```
- Signal generation on graph (SCAT-SW, SCAT-DW, SCAT-SN, SCAT-DN)
```
python trainFashion.py -s S [S/D] -g W [W/N]
```
- Graph and signal generation (SCAT-SW, SCAT-DW, SCAT-SN, SCAT-DN)
```
python -W ignore trainQM9.py -s S [S/D] -g W [W/N]
```

### Prerequisites

* Python >= 3.6
* Tensorflow >= 1.2.0
* RDKit
* networkx
* scipy, numpy, scikit-learn

## Citation
```
[1] Zou, D., & Lerman, G. (2018). Graph Convolutional Neural Networks via Scattering. Applied and Computational Harmonic Analysis, in press (available online 13 June 2019), DOI: 10.1016/j.acha.2019.06.003.
[2] Zou, D., & Lerman, G. (2019). Encoding Robust Representation for Graph Generation. International Joint Conference on Neural Networks (IJCNN) 2019.
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

We borrowed data and codes from various repositories we cited in [1].
