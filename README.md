

<p align="center"><img src="https://github.com/HongxinXiang/BenchMol/blob/master/assets/benchmol-icon.png" alt="logo" width="200px" /></p>

<h3 align="center">A Multi-Modality Benchmarking Platform For Molecular Representation</h3>

<h4 align="center">Supporting ADMET, QSAR, Virtual Screening and More</h4>

<div align="center">Built on PyTorch, benchmol is an easy-to-use and extensible Python package for drug discovery.</div>



<p align="center"><img src="https://github.com/HongxinXiang/BenchMol/blob/master/assets/framework.png" alt="BenchMol framework" width="90%" /></p>


---

## Table of Contents

- [News](#news)
- [Key Features](#key-features)
- [Installation](#installation)
- [Benchmarks](#benchmarks)
  - [MBANet](#mbanet)
  - [StructNet](#structnet)
- [Tutorials](#tutorials)
  - [Feature Extraction](#feature-extraction)
  - [Linear Probing](#linear-probing)
  - [Fine-tuning](#fine-tuning)
  - [Bench3M and Bench4M](#bench3m-and-bench4m)
- [Releases](#releases)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---


## News

- [2024/09/17] benchmol is released on GitHub.


---

## Key Features

- **Task coverage:** property prediction, ADMET, QSAR, virtual screening, and more.
- **Multi-modality support:** fingerprint, sequence, graph, geometry graph, image, geometry image, and video.
- **Broad baseline coverage:** a large collection of strong baselines for different molecular modalities.
- **New benchmarks:** **MBANet** and **StructNet**.

---


# Installation

### From PyPi

```bash
conda create --name test_env python=3.9
pip install benchmol
```

If some dependencies are not installed automatically, install them via:

```bash
pip install -r requirements_pip.txt
```

> Tip: If you encounter CUDA / PyTorch issues, please install a matching PyTorch version first, then install BenchMol.

---


# Benchmarks

BenchMol provides two benchmarks: **MBANet** and **StructNet**.


### MBANet

| Name   | Link                                                         | Description                                                  |
| ------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| MBANet | [OneDrive](https://1drv.ms/f/c/53030532e7d1aed6/EkBJ5uQDm8ROn6v1V8W3N6QBM724K5y0xB5ZSjkq0rg8qg?e=71m9XB) | Complete MBANet data, including csv files of molecules and data of different modalities processed |



### StructNet

| Name                        | Link                                                         | Description                                                  |
| --------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Raw Data                    | [OneDrive](https://1drv.ms/x/c/53030532e7d1aed6/EY6PJMH6N-9DnGfTcIiL4yIBf2IQaRcO5bPOw4jGQ2HwKQ?e=vcPKj1) | > 10 million molecules from CHEMBL 34                        |
| Raw Data of StructNet       | [OneDrive](https://1drv.ms/f/c/53030532e7d1aed6/Eq40uAkAGLlGmzS4hrwqAn8ByuFMv9mUIVGkpcNVEIfDsw?e=d7Yjrl) | The original data of StructNet includes csv files of 60 datasets. |
| Processed Data of StructNet | [OneDrive](https://1drv.ms/u/c/53030532e7d1aed6/EXRGFKBXVCRLhaHS2hKenfEBsVT2Qb3_Ao8ipALK6b7ksw?e=NtbEqP) | Processed data, including geometry, image, video and other multi-modal data |



# Tutorials

We provide tutorial notebooks demonstrating typical BenchMol workflows.



## Feature Extraction

BenchMol supports feature extraction from a wide range of molecular modalities.  
The following notebooks show how to generate features that can later be used for fusion, probing, or downstream training.

| Description                                                  | Tutorial Links                                               |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Extracting Molecular Fingerprints                            | [1_extract_fp_features.ipynb](https://github.com/HongxinXiang/BenchMol/blob/master/tutorials/1_extract_fp_features.ipynb) |
| Extracting features from sequence using un-pretrained CHEM-BERT | [1_extract_sequence_features.ipynb](https://github.com/HongxinXiang/BenchMol/blob/master/tutorials/1_extract_sequence_features.ipynb) |
| Extracting features from geometry image using IEM            | [1_extract_geometry_image_features.ipynb](https://github.com/HongxinXiang/BenchMol/blob/master/tutorials/1_extract_geometry_image_features.ipynb) |
| Extracting features from graph using GIN                     | [1_extract_graph_features.ipynb](https://github.com/HongxinXiang/BenchMol/blob/master/tutorials/1_extract_graph_features.ipynb) |
| Extracting features from molecular image using ImageMol      | [1_extract_image_features.ipynb](https://github.com/HongxinXiang/BenchMol/blob/master/tutorials/1_extract_image_features.ipynb) |
| Extracting features from video using VideoMol                | [1_extract_video_features.ipynb](https://github.com/HongxinXiang/BenchMol/blob/master/tutorials/1_extract_video_features.ipynb) |

The extracted outputs follow a unified format and can be plugged directly into the Bench3M / Bench4M pipelines below.


## Linear Probing

A complete linear probing example is provided here:  
- [2_linear_probing.ipynb](https://github.com/HongxinXiang/BenchMol/blob/master/tutorials/2_linear_probing.ipynb)



## Fine-tuning

Fine-tuning examples for different modalities are provided below:

| Description                              | Tutorial Links                                                        |
|------------------------------------------|-----------------------------------------------------------------------|
| Fine-tuning with sequence modality       | [3_fine_tuning_sequence.ipynb](https://github.com/HongxinXiang/BenchMol/blob/master/tutorials/3_fine_tuning_sequence.ipynb) |
| Fine-tuning with graph modality          | [3_fine_tuning_graph.ipynb](https://github.com/HongxinXiang/BenchMol/blob/master/tutorials/3_fine_tuning_graph.ipynb)    |
| Fine-tuning with geometry graph modality | [3_fine_tuning_geometry.ipynb](https://github.com/HongxinXiang/BenchMol/blob/master/tutorials/3_fine_tuning_geometry.ipynb)    |
| Fine-tuning with image modality          | [3_fine_tuning_image.ipynb](https://github.com/HongxinXiang/BenchMol/blob/master/tutorials/3_fine_tuning_image.ipynb)    |
| Fine-tuning with geometry image modality | [3_fine_tuning_geometry_image.ipynb](https://github.com/HongxinXiang/BenchMol/blob/master/tutorials/3_fine_tuning_geometry_image.ipynb)    |
| Fine-tuning with geometry video modality | [3_fine_tuning_video.ipynb](https://github.com/HongxinXiang/BenchMol/blob/master/tutorials/3_fine_tuning_video.ipynb)    |


## Bench3M and Bench4M

We provide **pre-extracted multi-modal features** for Bench3M and Bench4M,
so users can directly reproduce the training and evaluation results without
running the often time-consuming feature extraction stage.

The features can be downloaded from:

👉 **[Bench3M-4M Features (MoleculeNet)](https://1drv.ms/u/c/53030532e7d1aed6/IQD0l64eeP4TR5i_95iq9m7YAdWEs7So1ln2pFiFjc30I8U?e=WpHWfv)**

After downloading, place the feature files in your preferred data directory.
The tutorial notebooks allow you to specify the paths when constructing the dataset.

#### Running Examples

The following notebooks demonstrate how to train fusion / linear probing models using the provided features.

| Description         | Tutorial Links                                                                                    |
|---------------------|---------------------------------------------------------------------------------------------------|
| **Bench3M**         | [4_Bench3M.ipynb](https://github.com/HongxinXiang/BenchMol/blob/master/tutorials/4_Bench3M.ipynb) |
| **Bench4M**         | [4_Bench4M.ipynb](https://github.com/HongxinXiang/BenchMol/blob/master/tutorials/4_Bench4M.ipynb) |

Each tutorial includes:

- loading pre-extracted features  
- building the fusion / probing architecture  
- training and validation  
- evaluation and prediction saving  

These pipelines can be easily adapted to custom datasets or newly extracted features.

---

#### Using Your Own Features

If you prefer not to use the pre-computed features, you can generate them
with BenchMol using the feature extraction workflows described above.


# Releases

For version history and changelogs, see:  
- [Releases page](https://github.com/HongxinXiang/BenchMol/blob/master/RELEASE.md).


# Reference

If you find BenchMol useful, please consider citing our work:

```bibtex

```



## Acknowledgements

We thank the open-source community for the excellent molecular ML toolchains and datasets that BenchMol builds upon.

