# TNTSPA (Translating Natural language To SPARQL)

SPARQL is a highly powerful query language for an ever-growing number of Linked Data resources and Knowledge Graphs. Using it requires a certain familiarity with the entities in the domain to be queried as well as expertise in the language's syntax and semantics, none of which average human web users can be assumed to possess. To overcome this limitation, automatically translating natural language questions to SPARQL queries has been a vibrant field of research. However, to this date, the vast success of deep learning methods has not yet been fully propagated to this research problem. 

This paper contributes to filling this gap by evaluating the utilization of eight different Neural Machine Translation (NMT) models for the task of translating from natural language to the structured query language SPARQL. While highlighting the importance of high-quantity and high-quality datasets, the results show a dominance of a CNN-based architecture with a BLEU score of up to 98 and accuracy of up to 94%. 




## Research Paper

*Title: Neural Machine Translating from Natural Language to SPARQL*

*Authors: [Dr. Dagmar Gromann](http://dagmargromann.com/), [Prof. Sebastian Rudolph](http://sebastian-rudolph.de/doku.php?id=home) and [Xiaoyu Yin](https://www.linkedin.com/in/xiaoyu-yin-387966125/)*

> PDF is [available](https://arxiv.org/pdf/1906.09302.pdf)

```
@article{DBLP:journals/corr/abs-1906-09302,
  author    = {Xiaoyu Yin and
               Dagmar Gromann and
               Sebastian Rudolph},
  title     = {Neural Machine Translating from Natural Language to {SPARQL}},
  journal   = {CoRR},
  volume    = {abs/1906.09302},
  year      = {2019},
  url       = {http://arxiv.org/abs/1906.09302},
  archivePrefix = {arXiv},
  eprint    = {1906.09302},
  timestamp = {Thu, 27 Jun 2019 18:54:51 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1906-09302.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```



## Master Thesis

*Title: Translating Natural language To SPARQL*

*Author: Xiaoyu Yin*

*Supervisor: [Dr. Dagmar Gromann](http://dagmargromann.com/), [Dr. Dmitrij Schlesinger](https://cvl.inf.tu-dresden.de/people/dmitrij-schlesinger/)*

**The thesis is already finished. (8th January 2019)** and has been turned into a paper (link above).

> Find the thesis in [thesis folder](thesis), and defense slides in [presentation folder](presentation), both available in `.tex` and `.pdf` version. 



## Datasets

### Downloads ([Google drive]((https://drive.google.com/drive/folders/1V6c-y6tefKdZ4DfrNWhi_rlrDOngT1lS?usp=sharing)))
- [Monument](https://drive.google.com/drive/folders/1ibgd3pGtQZJ8lPTOCJ7vf6lzz2MxKa-0?usp=sharing)
- [Monument80](https://drive.google.com/drive/folders/18QF3avTHU8rD9C-hWAnD56QlP4yxhKDy?usp=sharing)
- [Monument50](https://drive.google.com/drive/folders/1C-vFYKpEvxCN06bjUvrqZBUb7hXeM145?usp=sharing)
- [LC-QUAD](https://drive.google.com/drive/folders/1LGk7a5aRKFQXWVsrdISz3jzzRD5TcdWb?usp=sharing)
- [DBNQA](https://drive.google.com/drive/folders/1sSiwVn7aBUezYvM4u226zzq5MqhbaxIw?usp=sharing)

### Usages

The files ended with `*.en` (e.g. `dev.en`, `train.en`, `test.en`) contain English sentences, `*.sparql` files contain SPARQL queries. The ones with the same prefix name have 1-1 mapping that was used in the training as a English-SPARQL pair. `vocab.*` or `dict.` are vocabulary files. [fairseq](https://github.com/facebookresearch/fairseq) has its own special requirement of input files, therefore aforementioned files were not used directly by it but processed into binary formats stored in `/fairseq-data-bin` folder of each dataset.

### Sources

The datasets used in this paper were originally downloaded from Internet. I downloaded them and have split them into the way I needed to train the models. The sources are listed as follows:
- [Neural SPARQL Machines Monument dataset](https://github.com/AKSW/NSpM/blob/master/data/monument_600.zip)
- [LC-QUAD](http://lc-quad.sda.tech/lcquad1.0.html) ([v2.0](http://lc-quad.sda.tech/index.html) is released! but we used 1.0)
- [DBpedia Neural Question Answering (DBNQA) dataset](https://figshare.com/articles/Question-NSpM_SPARQL_dataset_EN_/6118505)



## Experimental Setup

### Dataset splits and hyperparameters
see in paper

### Hardware configuration
![hardware](visualizations/hardware.png)

## Results


### Raw data

We kept the inference translations of each model and dataset which was used to generate BLEU scores, accuracy, and corresponding graphs in below sections. The results are saved in the format of `dev_output.txt` (validation set) & `test_output.txt` (test set) version and available [here (compat version)](results).
> [Full version](results_raw) containing raw output of frameworks is also available

### Training

> Plots of training perplexity for each models and datasets are available in a separate PDF [here](visualizations/graphs_perplexity_alldataset.pdf).

### Test results

Table of BLEU scores for all models and validation and test sets
![Bleu scores](visualizations/best-bleu-scores.png)

Table of Accuracy (in %) of syntactically correct generated SPARQL queries | F1 score
![accuracy](visualizations/accuracy-sparql-queries.png)

> Please find more results and detailed explanations in the research paper and the thesis.


## Trained Models

Because some models were so space-consuming (esp. GNMT4, GNMT8) after training for some sepecific datasets (esp. DBNQA), I didn't download all the models from the HPC server. This is an overview of the availablity of the trained models on [my drive](https://drive.google.com/drive/folders/1VuZrbFl3hgK-qWwGV_zI68qtZWKAKbTv?usp=sharing):

. | Monument | Monument80 | Monument50 | LC-QUAD | DBNQA
-- | -- | -- | -- | -- | --
NSpM | [yes](https://drive.google.com/drive/folders/1Shb58SQIrmXiXStHMemRoNRineZUFFO-?usp=sharing) | [yes](https://drive.google.com/drive/folders/1c1aoLH8rkOYYUW_CQ12jhxWFnj_M7LWe?usp=sharing) | [yes](https://drive.google.com/drive/folders/1metc-Ma9bumdDCNbpgHOqyx3IYOXmKP3?usp=sharing) | [yes](https://drive.google.com/drive/folders/10kN_gdSDaLnJWfBC8-kZPpr2VZwmHP87?usp=sharing) | [yes](https://drive.google.com/drive/folders/1b55dwI6w2OEOirOrnppy1YXIWpqn1QOm?usp=sharing)
NSpM+Att1 | [yes](https://drive.google.com/drive/folders/1E8gZ_eL-b4qf-Jog0tFT7gd1d8oU6vTi?usp=sharing) | [yes](https://drive.google.com/drive/folders/11M0HXt6YC8FKZKmgEeGoseQ7E4deSXp9?usp=sharing) | [yes](https://drive.google.com/drive/folders/1yMWLF0hSkeBEOSh1QI7Vme36lGAvig0H?usp=sharing) | [yes](https://drive.google.com/drive/folders/18vF_FWKRboHUmfDLIg90CYMzyy2OGo3c?usp=sharing) | [yes](https://drive.google.com/drive/folders/1r4vcHSqQlplrlcAERiMBS1BvAz7_yrUs?usp=sharing)
NSpM+Att2 | [yes](https://drive.google.com/drive/folders/13X8yPV_2SRF7YzrwQI2Kymj7zTW2SDWR?usp=sharing) | [yes](https://drive.google.com/drive/folders/1uaTuBJS838kXVtxEEGC2n2winwu-4PyU?usp=sharing) | [yes](https://drive.google.com/drive/folders/1KPcbwqo_G00hUodKQhNi0K4MLo73zzOR?usp=sharing) | [yes](https://drive.google.com/drive/folders/1Ohkq58_D9gSYdgZjRe4kwDf2EtJvCgJ0?usp=sharing) | [yes](https://drive.google.com/drive/folders/1iV_GfAhBvtVjog4yspw_JfW4eePdhJD7?usp=sharing)
GNMT4 | no | [yes](https://drive.google.com/drive/folders/14xi_4LYL1PD-WD_FPqJLzzY4jgDgHFYx?usp=sharing) | no | no | no
GNMT8 | no | no | no | no | no
LSTM_Luong | [yes](https://drive.google.com/drive/folders/1KrEZIpE80lxBIMoV6r3J8G8xtaq27z_F?usp=sharing) | [yes](https://drive.google.com/drive/folders/1Asj8WCtcZcC8M58jIbrVRMwTGChtVVmy?usp=sharing) | [yes](https://drive.google.com/drive/folders/1pvnOPMKfYXLE6a99vy5J5_oettA_X4CI?usp=sharing) | [yes](https://drive.google.com/drive/folders/1XmL58DBjIpTUdfldZr7TPn2U5YGs7tPX?usp=sharing) | no
ConvS2S | [yes](https://drive.google.com/drive/folders/1-PlqdxH6FlZckGkPWf_QLeSEaluHq81D?usp=sharing) | [yes](https://drive.google.com/drive/folders/19DvUk_Lh_rRYxSa9I7mb3dUAEHB4_1zA?usp=sharing) | [yes](https://drive.google.com/drive/folders/1NRiMMXCN9shMB25ZXqabVf8KOadvJ4GY?usp=sharing) | [yes](https://drive.google.com/drive/folders/1DBE4aOSyyuen4fU1eYo3QCDpOKNDuvZO?usp=sharing) | no
Transformer | [yes](https://drive.google.com/drive/folders/1KrnbRQvwbMSx1lMvqn4k27ISmaqbqvki?usp=sharing) | [yes](https://drive.google.com/drive/folders/1vWM0UuKZlcSvz-fRk1GnFMKGJPq9mIFL?usp=sharing) | [yes](https://drive.google.com/drive/folders/172mg0sMNg-vOiiya26eaYT_LblMcoe-u?usp=sharing) | [yes](https://drive.google.com/drive/folders/1H8_qjb6Aa_YrOyL-WI6dabegSXNSFNTh?usp=sharing) | no



## One More Thing

This paper and thesis couldn't have been completed without the help of my supervisors ([Dr. Dagmar Gromann](http://dagmargromann.com/), [Dr. Dmitrij Schlesinger](https://cvl.inf.tu-dresden.de/people/dmitrij-schlesinger/) and [Prof. Sebastian Rudolph](http://sebastian-rudolph.de/doku.php?id=home)) and those great open source projects. I send my sincere appreciation to  all of the people who have been working on this subject, and hopefully we will show the world its value in the near future.

- [Neural SPARQL Machines](https://github.com/AKSW/NSpM)
- [LC-QUAD](http://lc-quad.sda.tech/index.html)
- [DBNQA](https://github.com/AKSW/DBNQA)
- [fairseq](https://github.com/facebookresearch/fairseq)
- [nmt](https://github.com/tensorflow/nmt)

By the way, I work as an Android developer now, although I still have passion with AI and may want to learn more and probably even find a career in it in the future, currently my focus is on Software Engineering. I enjoy any kind of experience or knowledge sharing and would like to have new friends! Connect with me on [LinkedIn](https://www.linkedin.com/in/xiaoyu-yin-387966125/). 
