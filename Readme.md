# Introduction

This repository is the formal implement of our paper titled “[Robust Domain Misinformation Detection via Multi-modal Feature Alignment](https://ieeexplore.ieee.org/abstract/document/10288548/)”. The contribution of this work can be summarized as follows:

1. A unified framework that tackles the domain generalization (target domain data is unavailable) and domain adaptation tasks (target domain data is available). This is necessary as obtaining sufficient unlabeled data in the target domain at an early stage of misinformation dissemination is difficult.
2. Inter-domain and cross-modality alignment modules that reduce the domain shift and the modality gap. These modules aim at learning rich features that allow misinformation detection. Both modules are plug-and-play and have the potential to be applied to other multi-modal tasks.

Additionally, we believe that the multimodal generalization algorithms proposed in our work can be used in other multimodal tasks. If you have some questions related to this paper, please feel no hesitate to ask me. 

# To run our code

1. download the dataset and pretrained models from Onedrive([datasets_pretrain_models.zip](https://portland-my.sharepoint.com/:u:/g/personal/liuhui3-c_my_cityu_edu_hk/ESqS-qpzsYJGoawSrzKDOlcBWr5e2plKA47L1JX0zsF4Ug?e=Gqu5Qi)) and unzip them in  this project dir.

2. drive_outmodel.py is the main file to drive our algorithms. Please remove the codes related to comel package that enable efficient management of ML experiments or add your api_key and other parameters in the below codes in this file:

   ```python
    experiment = Experiment(
           api_key="",
           project_name="",
           workspace="",
       )
   ```

3. Multimodal JMMD, in our work, devised for multimodal  generalization tasks, can capture cross-modal correlations among multiple modalities with theoretical guarantees. For better implement, I advise to use the implement of MMD in domainbed that fix the parameter of kernels and only adjust the weight of JMMD loss function $\lambda_1$. Otherwise, you can just use my implement to set the kernel manually.

4. At last, you can run our codes as below:

   ```
   sh multi_out_model.sh
   ```

# Citation

If you find this repository helpful, please cite our paper:

```
@ARTICLE{10288548,
  author={Liu, Hui and Wang, Wenya and Sun, Hao and Rocha, Anderson and Li, Haoliang},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={Robust Domain Misinformation Detection via Multi-modal Feature Alignment}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TIFS.2023.3326368}}
```

If you have interest in multimodal misinformation detection, another paper of me on multimodal misinformation task can help you https://arxiv.org/abs/2305.05964. 

```
@inproceedings{DBLP:conf/acl/LiuWL23,
  author       = {Hui Liu and
                  Wenya Wang and
                  Haoliang Li},
  editor       = {Anna Rogers and
                  Jordan L. Boyd{-}Graber and
                  Naoaki Okazaki},
  title        = {Interpretable Multimodal Misinformation Detection with Logic Reasoning},
  booktitle    = {Findings of the Association for Computational Linguistics: {ACL} 2023,
                  Toronto, Canada, July 9-14, 2023},
  pages        = {9781--9796},
  publisher    = {Association for Computational Linguistics},
  year         = {2023},
  url          = {https://doi.org/10.18653/v1/2023.findings-acl.620},
  doi          = {10.18653/V1/2023.FINDINGS-ACL.620},
  timestamp    = {Thu, 10 Aug 2023 12:35:42 +0200},
  biburl       = {https://dblp.org/rec/conf/acl/LiuWL23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
