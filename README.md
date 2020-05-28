# AHAG

This is our implementation for the paper:


*Adaptive Hierarchical Attention-Enhanced Gated Network Integrating Reviews for Item Recommendation*



AHAG: This is the state-of-the-art method that uti-lizes deep learning technology to jointly model user and item from reviews for item recommendation.


## Environments

- python 3.5
- Tensorflow (version: 1.9.0)
- numpy
- pandas


## Dataset

In our experiments, we use the datasets from  Amazon 5-core(http://jmcauley.ucsd.edu/data/amazon) 
Pretrained [GloVe embeddings](https://nlp.stanford.edu/projects/glove/) obtained from Wikipedia 2014 + Gigaword 5 with 6B tokens used for words.

## Example to run the codes		

Data preprocessing:

The implemention of data preprocessing is modified based on *[this](https://github.com/chenchongthu/DeepCoNN)*


Train and evaluate the model:

```
python train.py
```



## Misc
The implemention of CNN is modified based on *[this](https://github.com/chenchongthu/NARRE)*
The implemention of self-attention is modified based on *[this](https://github.com/Kyubyong/transformer)*




