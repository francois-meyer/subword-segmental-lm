# Subword Segmental Language Modelling for Nguni Languages

Paper: Subword Segmental Language Modelling for Nguni Languages

Francois Meyer and Jan Buys, Findings of EMNLP 2022

Train subword segmental language models (SSLMs) - language models that learn how to segment words while being trained for autoregressive language modelling.

Evaluate trained SSLMs on intrinsic language modelling performance (BPC) and as unsupervised morphological segmenters.

The datasets and models produced for this paper are publicly available:
* [Nguni language modelling datasets](https://drive.google.com/file/d/1be7y9LWIpDkx2UTe7xYTkZ0TWi3VQ3LQ/view?usp=sharing) - train/validation/test splits for isiXhosa, isiZulu, isiNdebele, and Siswati.
* [Trained Nguni subword segmental language models](https://drive.google.com/file/d/1UeifY5ttCaygCYWVqe21TE8wYkvqYqyR/view?usp=sharing)

### Dependencies (Python) ###

* NumPy
* PyTorch
* Torchtext
* tqdm
* NLTK
