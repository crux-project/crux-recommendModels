# This is the top-k model recommendation in the inductive setting on the graph.

The majority of the code comes from the paper:

Yunfan Wu, Qi Cao, Huawei Shen, Shuchang Tao & Xueqi Cheng. 2022. **INMO: A Model-Agnostic and Scalable Module for Inductive
Collaborative Filtering**  , In *Proceedings of SIGIR'22*.

Thanks for the authors releasing their codes

## Environment

Python 3.8

Pytorch >= 1.8

DGL >= 0.8

## Dataset

The processed data is the Crux dataset.
The raw dataset is in another project-hosted dataset:
crux-project/CRUX/gnn/input

The processing parser is
crux-project/crux-recommendModels/run/dataPrep/parser.py
