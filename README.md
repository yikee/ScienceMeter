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
The `eval_judgment.py` and `eval_generation.py` scripts are used to evaluate a specific type of scientific knowledge in the `model`, assumed to be a knowledge-updated version of the `basemodel`. If the `model` and `basemodel` are the same, the evaluation is performed on the `basemodel`. The `--portion` argument allows control over the fraction of the dataset used for evaluation.

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
  --model _ar_testdoc \
  --domain education \
  --knowledge future \
  --portion 1.0
```
## Evaluation of knowledge Update Methods
The `metrics.py` script computes all eight evaluation metrics introduced in the paper, based on evaluation results obtained before (`basemodel`) and after (`model`) a knowledge update. The `model` is assumed to be a knowledge-updated version of the `basemodel`, using a specified update method (e.g., `_ar_traintestdoc_it_trainqa`).

```bash
# example
python metrics.py \
  --basemodel llama \
  --model _ar_traintestdoc_it_trainqa \
  --domain political_science \
  --task judgment
```

## Questions
If you have any questions or comments about our paper, data, or scripts, or if you notice any issues in the code, feel free to reach out via email at `yikewang@cs.washington.edu`. We will do our best to respond within one business day.

## Citing
If you found this work helpful, please consider starring this repository and citing our paper as shown below:
```latex
@article{wang2025sciencemeter,
  title={ScienceMeter: Tracking Scientific Knowledge Updates in Language Models},
  author={Wang, Yike and Feng, Shangbin and Tsvetkov, Yulia and Hajishirzi, Hannaneh},
  journal={arXiv preprint arXiv:2505.24302},
  year={2025}
}
