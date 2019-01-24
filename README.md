# Hybrid Attention-Based Prototypical Networks for Noisy Few-Shot Relation Classification

Code and data for AAAI2019 paper [Hybrid Attention-Based Prototypical Networks for Noisy Few-Shot Relation Classification](https://gaotianyu1350.github.io/assets/aaai2019_hatt_paper.pdf).

Author: Tianyu Gao*, Xu Han*, Zhiyuan Liu, Maosong Sun. (\* means equal contribution)

## Dataset and Word Embedding

We evaluate our models on [FewRel](https://thunlp.github.io/fewrel), a large-scale dataset for few-shot relation classification. It has 100 relations and 700 instances for each relation. You can find some baseline models from [here](https://github.com/thunlp/fewrel).

Due to the large size, we did not upload the glove file (pre-trained word embedding). Please download `glove.6B.50d.json` from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/b14bf0d3c9e04ead9c0a/?dl=1) or [Google Drive](https://drive.google.com/open?id=1UnncRYzDpezPkwIqhgkVW6BacIqz6EaB) and put it under `data/` folder.

## Usage

To run our code, use this command for training
```bash
python train.py {MODEL_NAME} {N} {K} {NOISE_RATE}
```
and use this command for testing
```bash
python test.py {MODEL_NAME} {N} {K} {NOISE_RATE}
```
where {MODEL_NAME} could be `proto` or `proto_hatt`, `{N}` is the num of classes, `{K}` is the num of instances for each class and `{NOISE_RATE}` is the probability that one instance is wrong-labeled.
