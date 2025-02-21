# Automated Phase Mapping of XRD (AutoMapper)

The AutoMapper is an unsupervised solver to tackle the phase mapping challenge in high-throughput X-ray diffraction datasets. Besides leveraging robust fitting abilities of machine learning algorithms,it is integrated various material information, including first-principles calculated thermodynamic data, crystallography, X-ray diffraction, and texture.

Its main functionalities:

- correctly identify the number, identity, fraction, peak shift and texture of present phases. 


<p align="center">
  <img src="phasemapy/Overview of AutoMapper.svg" /> 
</p>



## Table of Contents

- [Installation](#installation)
- [Datasets](#datasets)
- [Citation](#citation)
- [Contact](#contact)

## Installation


Run the following command to install the environment:
```
pip install -r environment.yaml
```


## Datasets

We analyzed three previously published combinatorial libraries: the V–Nb–Mn oxide, the Bi–Cu–V oxide, and the Li–Sr–Al oxide systems, which contain 317, 307, and 50 samples, respectively. If you use these datasets, please consider to cite the original papers from which we curate these datasets.



## Citation

Please consider citing the following paper if you find our code & data useful.

```
@article{XXX,
  title={Automated Phase Mapping of High Throughput X-ray Diffraction Data Encoded with Domain-Specific Materials Science Knowledge},
  author={Dongfang Yu, Sean D. Griesemer, Tzu-chen Liu, Chris Wolverton, Yizhou Zhu},
  journal={XXX},
  year={2024}
}
```

## Contact

Please leave an issue or reach out to Yizhou Zhu (zhuyizhou AT westlake DOT edu DOT cn ) if you have any questions.
