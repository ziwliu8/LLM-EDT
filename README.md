# LLM-EDT: Large Language Model Enhanced Cross-domain Sequential Recommendation with Dual-phase Training

This is the implementation of the paper "LLM-EDT: Large Language Model Enhanced Cross-domain Sequential Recommendation with Dual-phase Training".

## Configure the environment

To ease the configuration of the environment, I list versions of my hardware and software equipments:

- Hardware:
  - GPU: RTX 4090 24GB
  - Cuda: 12.0
- Software:
  - Python: 3.11.16
  - Pytorch: 2.4.1

You can pip install the `requirements.txt` to configure the environment.

## Preprocess the dataset

You can preprocess the dataset and get the LLMs embedding according to the following steps:

1. The raw dataset downloaded from the website should be put into `/data/<amazon/elec/douban>/raw/`. Specifically, The Cloth-Sport and Electronics - Cell Phone and Food-Kitchen datasets can be obtained from this [Download Link](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/).
2. Conduct the preprocessing ipynb `data/<amazon/elec/food>/` to process the data augmentation for cross-domain sequential recommendation. After the procedure, you will get the id file  `/data/<amazon/elec/douban>/handled/aug_id_map.json` and the interaction file  `/data/<amazon/elec/douban>/handled/<aug_cloth-sport/aug_elec-phone/aug_food_kitchen>.pkl` , the LLMs item embedding file `/data/<<amazon/elec/douban>/handled/aug_itm_emb_np.pkl`, and the unbiased user profile embedding: `domain_split_usr_profile_emb.pkl`.

In conclusion, the prerequisite files to run the code are as follows: `aug_<cloth-sport/elec-phone/book-movie>.pkl`, `aug_itm_emb_np_all.pkl`, `domain_split_usr_profile_emb.pkl` and `aug_id_map.json`.

⭐️ To ease the reproducibility of our paper, we also upload all preprocessed files to this [link](https://ufile.io/tylmbg40).
The well-trained weight file can be found in this [link](https://ufile.io/gku0ngze).
## Run and test

1. You can reproduce all LLM-EDT experiments by running the bash step by step as follows:
Global Pretraining Stage:
```
bash experiments/<amazon/elec/food>/one4all.bash
```
Domain Fine-tuning Stage:
```
bash experiments/<amazon/elec/food>/domain_adapter.bash
```

2. The log and results will be saved in the folder `log/`. The checkpoint will be saved in the folder `saved/`.
