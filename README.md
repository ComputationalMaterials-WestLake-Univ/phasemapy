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
- [Usage](#usage)
- [Authors and acknowledgements](#authors-and-acknowledgements)
- [Citation](#citation)
- [Contact](#contact)

## Installation

The easiest way to install prerequisites is via [conda](https://conda.io/docs/index.html).

Run the following command to install the environment:
```
pip install -r requirements.txt
```


## Datasets

We analyzed three previously published combinatorial libraries: the V–Nb–Mn oxide, the Bi–Cu–V oxide, and the Li–Sr–Al oxide systems, which contain 317, 307, and 50 samples, respectively. If you use these datasets, please consider to cite the original papers from which we curate these datasets.

Find more about these datasets by going to our [Datasets_Bi_Cu_V_O]("phasemapy/scripts_Bi_Cu_V_O/data) page,  [Datasets_V_Nb_Mn_O]("phasemapy/scripts_V_Nb_Mn_O/data) page.,  [Datasets_Li_Sr_Al_O]("phasemapy/scripts_Li_Sr_Al_O/data) page.

## Usage
### Phase mapping with V–Nb–Mn dataset:

To solve the V–Nb–Mn dataset, run the following command:

```
python phasemapy/scripts_V_Nb_Mn_O/solver_V-Nb-Mn.py
```
### Phase mapping with Bi-Cu-V dataset:

To solve the Bi-Cu-V dataset, run the following command:

```
python phasemapy/scripts_Bi_Cu_V_O/solver_Bi_Cu_V.py
```

### Phase mapping with Li–Sr–Al dataset:

To solve the Li–Sr–Al dataset, run the following command:

```
python phasemapy/scripts_Li_Sr_Al_O/solver_Li_Sr_Al.py
```


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
