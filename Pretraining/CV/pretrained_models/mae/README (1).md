---
license: apache-2.0 tags:

- vision datasets:
- imagenet-1k

---

# Vision Transformer (base-sized model) pre-trained with MAE

Vision Transformer (ViT) model pre-trained using the MAE method. It was introduced in the
paper [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) by Kaiming He, Xinlei Chen,
Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick and first released
in [this repository](https://github.com/facebookresearch/mae).

Disclaimer: The team releasing MAE did not write a model card for this model so this model card has been written by the
Hugging Face team.

## Model description

The Vision Transformer (ViT) is a transformer encoder model (BERT-like). Images are presented to the model as a sequence
of fixed-size patches.

During pre-training, one randomly masks out a high portion (75%) of the image patches. First, the encoder is used to
encode the visual patches. Next, a learnable (shared) mask token is added at the positions of the masked patches. The
decoder takes the encoded visual patches and mask tokens as input and reconstructs raw pixel values for the masked
positions.

By pre-training the model, it learns an inner representation of images that can then be used to extract features useful
for downstream tasks: if you have a dataset of labeled images for instance, you can train a standard classifier by
placing a linear layer on top of the pre-trained encoder.

## Intended uses & limitations

You can use the raw model for image classification. See
the [model hub](https://huggingface.co/models?search=facebook/vit-mae) to look for fine-tuned versions on a task that
interests you.

### How to use

Here is how to use this model:

```python
from transformers import AutoFeatureExtractor, ViTMAEForPreTraining
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/vit-mae-base')
model = ViTMAEForPreTraining.from_pretrained('facebook/vit-mae-base')

inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
loss = outputs.loss
mask = outputs.mask
ids_restore = outputs.ids_restore
```

### BibTeX entry and citation info

```bibtex
@article{DBLP:journals/corr/abs-2111-06377,
  author    = {Kaiming He and
               Xinlei Chen and
               Saining Xie and
               Yanghao Li and
               Piotr Doll{\'{a}}r and
               Ross B. Girshick},
  title     = {Masked Autoencoders Are Scalable Vision Learners},
  journal   = {CoRR},
  volume    = {abs/2111.06377},
  year      = {2021},
  url       = {https://arxiv.org/abs/2111.06377},
  eprinttype = {arXiv},
  eprint    = {2111.06377},
  timestamp = {Tue, 16 Nov 2021 12:12:31 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2111-06377.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```