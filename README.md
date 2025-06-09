## ScienceMeter: Tracking Scientific Knowledge Updates in Language Models

<div align="center">
  <b>Yike Wang<sup>1</sup>, Shangbin Feng<sup>1</sup>, Yulia Tsvetkov<sup>1</sup>, Hannaneh Hajishirzi<sup>1</sup><sup>2</sup></b>
  <br>
  <sup>1</sup>University of Washington, <sup>2</sup>Allen Institute for Artificial Intelligence
  <br><br>
  <a href=""><img src="https://img.shields.io/badge/Paper-arXiv-orange"></a>
</div>

<p align="center">
  <img src="https://github.com/user-attachments/assets/2d52257d-9543-4064-943a-d5b5800d43d9" width="800" />
</p>


⚠️ This repository is currently a work in progress and is not yet complete. Code and documentation may be missing or subject to change. We will try to complete it as soon as possible. Thank you for your patience!

## Dataset
We retrieve 1,000 journal or conference papers from each of 10 scientific domains using the [Semantic Scholar API](https://www.semanticscholar.org/product/api#api-key-form). For each paper, we also collect its citing papers, forming our `raw` corpus.

We filter out papers that lack citation information or abstracts, then regroup the remaining papers based on the knowledge cutoff date of a given model and the publication dates of the papers. This process yields 5,148 triplets of (prior paper, new paper, future paper). For each paper, we synthetically generate one SUPPORT claim (a uniquely supporting scientific claim) and one REFUTE claim (a relevant but non-supporting scientific claim). The resulting dataset is available in the `filtered_with_claims` folder.

## Evaluation of Scientific Knowledge
The `eval_judgment.py` and `eval_generation.py` scripts evaluate a specific type of scientific knowledge in the `model`, which represents the knowledge-updated version of the `basemodel`. If `model` is the same as `basemodel`, the evaluation will be performed on the unupdated base model. The proportion of data to evaluate can be adjusted using the `--portion` argument.

### Claim Judgment Task
```bash
# example
python eval_judgment.py \
  --basemodel llama \
  --model llama \
  --domain computer_science \
  --knowledge new \
  --portion 0.8
```

### Claim Generation Task
```bash
# example
python eval_generation.py \
  --basemodel olmo32b \
  --model ar_testdoc \
  --domain education \
  --knowledge future \
  --portion 1.0
```

## Citing
If you found this work helpful, please consider starring this repository and citing our paper as shown below:
```latex
@article{wang2025sciencemeter,
  title={ScienceMeter: Tracking Scientific Knowledge Updates in Language Models},
  author={Wang, Yike and Feng, Shangbin and Tsvetkov, Yulia and Hajishirzi, Hannaneh},
  journal={arXiv preprint arXiv:2505.24302},
  year={2025}
}
