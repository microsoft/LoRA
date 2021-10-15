# LoRA: Low-Rank Adaptation of Large Language Models


This repo contains the source code of the Python package `loralib` and several examples of how to integrate it with PyTorch models, such as those in HuggingFace.
We only support PyTorch for now.
See our paper for a detailed description of LoRA.

**LoRA: Low-Rank Adaptation of Large Language Models** <br>
*Edward J. Hu\*, Yelong Shen\*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen* <br>
Paper: https://arxiv.org/abs/2106.09685 <br>

LoRA reduces the number of trainable parameters by learning pairs of rank-decompostion matrices while freezing the original weights.
This vastly reduces the storage requirement for large language models adapted to specific tasks and enables efficient task-switching during deployment all without introducing inference latency.
LoRA also outperforms several other adaptation methods including adapter, prefix-tuning, and fine-tuning.

We obtain result comparable or superior to full finetuning on the GLUE benchmark using [RoBERTa (Liu et al., 2019)](https://arxiv.org/abs/1907.11692) base and large and [DeBERTa (He et al., 2020)](https://arxiv.org/abs/2006.03654) XXL 1.5B, while only training and storing a fraction of the parameters. Click the numbers below to download the RoBERTa and DeBERTa LoRA checkpoints.

|   |         | RoBERTa base <br> Fine-tune  |  RoBERTa base <br> LoRA  | DeBERTa XXL <br> Fine-tune | DeBERTa XXL <br> LoRA  |
|---|-------------------------|----------------|--------------------------|-----------------|-----------------|
|   | # of Trainable Params.  | 125M | 0.8M | 1.5B | 4.7M     |
|   | MNLI (m-Acc/mm-Acc)     | <b>87.6</b> | [<b>87.5</b>±.3/86.9±.3](https://github.com/microsoft/LoRA/releases/download/RoBERTa-base/roberta_base_lora_mnli.bin) |91.7/<b>91.9</b>| [<b>91.9</b>±.1/<b>91.9</b>±.2](https://github.com/microsoft/LoRA/releases/download/DeBERTa/deberta_v2_xxlarge_lora_mnli.bin)       |
|   | SST2 (Acc)              | 94.8 | [<b>95.1</b>±.2](https://github.com/microsoft/LoRA/releases/download/RoBERTa-base/roberta_base_lora_sst2.bin) | <b>97.2</b>    | [96.9±.2](https://github.com/microsoft/LoRA/releases/download/DeBERTa/deberta_v2_xxlarge_lora_sst2.bin)                    |
|   | MRPC (Acc)              | <b>90.2</b> | [<b>89.7</b>±.7](https://github.com/microsoft/LoRA/releases/download/RoBERTa-base/roberta_base_lora_mrpc.bin) | 92.0           | [<b>92.6</b>±.6](https://github.com/microsoft/LoRA/releases/download/DeBERTa/deberta_v2_xxlarge_lora_mrpc.bin)             |
|   | CoLA (Matthew's Corr)   | <b>63.6</b> | [<b>63.4</b>±1.2](https://github.com/microsoft/LoRA/releases/download/RoBERTa-base/roberta_base_lora_cola.bin) | <b>72.0</b>    | [<b>72.4</b>±1.1](https://github.com/microsoft/LoRA/releases/download/DeBERTa/deberta_v2_xxlarge_lora_cola.bin)           |
|   | QNLI (Acc)              | 92.8 | [<b>93.3</b>±.3](https://github.com/microsoft/LoRA/releases/download/RoBERTa-base/roberta_base_lora_qnli.bin) | <b>96.0</b>    | [<b>96.0</b>±.1](https://github.com/microsoft/LoRA/releases/download/DeBERTa/deberta_v2_xxlarge_lora_qnli.bin)            |
|   | QQP (Acc)               | <b>91.9</b> | [90.8±.1](https://github.com/microsoft/LoRA/releases/download/RoBERTa-base/roberta_base_lora_qqp.bin) | 92.7           | [<b>92.9</b>±.1](https://github.com/microsoft/LoRA/releases/download/DeBERTa/deberta_v2_xxlarge_lora_qqp.bin)           |
|   | RTE (Acc)               | 78.7 | [<b>86.6</b>±.7](https://github.com/microsoft/LoRA/releases/download/RoBERTa-base/roberta_base_lora_rte.bin) | 93.9           | [<b>94.9</b>±.4](https://github.com/microsoft/LoRA/releases/download/DeBERTa/deberta_v2_xxlarge_lora_rte.bin)           |
|   | STSB (Pearson/Spearman Corr) | 91.2 | [<b>91.5</b>±.2/<b>91.3</b>±.2](https://github.com/microsoft/LoRA/releases/download/RoBERTa-base/roberta_base_lora_stsb.bin) |<b>92.9</b>/92.6| [<b>93.0</b>±.2/<b>92.9</b>±.3](https://github.com/microsoft/LoRA/releases/download/DeBERTa/deberta_v2_xxlarge_lora_stsb.bin)      |
|   | Average  | 86.40 | <b>87.24</b> | 91.06 | <b>91.32</b> |

<i>Note: You still need the original pre-trained checkpoint from [HuggingFace](https://huggingface.co/) to use the LoRA checkpoints.</i>

Fine-tuning numbers are taken from [Liu et al. (2019)](https://arxiv.org/abs/1907.11692) and [He et al. (2020)](https://arxiv.org/abs/2006.03654).  We include confidence intervals on results from our experiments. Please follow the instructions in `examples/NLU/` to reproduce our results.

On GPT-2, LoRA compares favorably to both full finetuning and other efficient tuning methods, such as [adapter (Houlsby et al., 2019)](https://arxiv.org/abs/1902.00751) and [prefix tuning (Li and Liang, 2021)](https://arxiv.org/abs/2101.00190). We evaluated on E2E NLG Challenge, DART, and WebNLG:

|   | Method              | # of Trainable Params | E2E (BLEU)   | DART (BLEU)  | WebNLG (BLEU-U/S/A)            |
|---|---------------------|-----------------------|--------------|--------------|--------------------------------|
|   | GPT-2 M (Fine-Tune) | 354.92M               | 68.2         | 46.0         | 30.4/<b>63.2</b>/47.6          |
|   | GPT-2 M (Adapter)   | 0.37M                 | 66.3         | 42.4         | 45.1/54.5/50.2                 |
|   | GPT-2 M (Prefix)    | 0.35M                 | 69.7         | 45.7         | 44.1/63.1/54.4                 |
|   | GPT-2 M (LoRA)      | 0.35M                 |<b>70.4</b>±.1|<b>47.1</b>±.2| <b>46.7</b>±.4/62.1±.2/<b>55.3</b>±.2 |
|   | GPT-2 L (Fine-Tune) | 774.03M               | 68.5         | 46.5         | 41.7/<b>64.6</b>/54.2          |
|   | GPT-2 L (Adapter)   | 0.88M                 | 69.1±.1      | 45.7±.1      | <b>49.8</b>±.0/61.1±.0/56.0±.0 |
|   | GPT-2 L (Prefix)    | 0.77M                 | 70.3         | 46.5         | 47.0/64.2/56.4                 |
|   | GPT-2 L (LoRA)      | 0.77M                 |<b>70.4</b>±.1|<b>47.5</b>±.1| 48.4±.3/<b>64.0</b>±.3/<b>57.0</b>±.1 |

Non-LoRA baselines, except for adapter on GPT-2 large, are taken from [Li and Liang (2021)](https://arxiv.org/abs/2101.00190). We include confidence intervals on results from our experiments.

Download the GPT-2 LoRA checkpoints:
 * [GPT-2 Medium E2E](https://github.com/microsoft/LoRA/releases/download/GPT-2/gpt2_md_lora_e2e.pt) (1.5 MB)
 * [GPT-2 Medium DART](https://github.com/microsoft/LoRA/releases/download/GPT-2/gpt2_md_lora_dart.pt) (1.5 MB)
 * [GPT-2 Medium WebNLG](https://github.com/microsoft/LoRA/releases/download/GPT-2/gpt2_md_lora_webnlg.pt) (1.5 MB)
 * [GPT-2 Large E2E](https://github.com/microsoft/LoRA/releases/download/GPT-2/gpt2_lg_lora_e2e.pt) (2.3 MB)
 * [GPT-2 Large DART](https://github.com/microsoft/LoRA/releases/download/GPT-2/gpt2_lg_lora_dart.pt) (2.3 MB)
 * [GPT-2 Large WebNLG](https://github.com/microsoft/LoRA/releases/download/GPT-2/gpt2_lg_lora_webnlg.pt) (2.3 MB)

Please follow the instructions in `examples/NLG/` to reproduce our result.
## Repository Overview

<i>(The initial release of this repo has been archived in the branch "snapshot-9-15-2021")</i>

There are several directories in this repo:
* [loralib/](loralib) contains the source code for the package `loralib`, which needs to be installed to run the examples we provide;
* [examples/NLG/](examples/NLG) contains an example implementation of LoRA in GPT-2 using our package, which can be used to reproduce the result in our paper;
* [examples/NLU/](examples/NLU) contains an example implementation of LoRA in RoBERTa and DeBERTa using our package, which produces competitive results on the GLUE benchmark;
* See how we use `loralib` in [GPT-2](examples/NLG/src/model.py), [RoBERTa](examples/NLU/src/transformers/models/roberta/modeling_roberta.py), and [DeBERTa v2](examples/NLU/src/transformers/models/deberta_v2/modeling_deberta_v2.py)

## Quickstart

 1. Installing `loralib` is simply
 ```
 pip install loralib
 # Alternatively
 # pip install git+https://github.com/microsoft/LoRA
 ```

 2. You can choose to adapt some layers by replacing them with counterparts implemented in `loralib`. We only support `nn.Linear`, `nn.Embedding`, and `nn.Conv2d` for now. We also support a `MergedLinear` for cases where a single `nn.Linear` represents more than one layers, such as in some implementations of the attention `qkv` projection (see Additional Notes for more).
 ```
 # ===== Before =====
 # layer = nn.Linear(in_features, out_features)

 # ===== After ======
 import loralib as lora
 # Add a pair of low-rank adaptation matrices with rank r=16
 layer = lora.Linear(in_features, out_features, r=16)
 ```

 3. Before the training loop begins, mark only LoRA parameters as trainable.
 ```
 import loralib as lora
 model = BigModel()
 # This sets requires_grad to False for all parameters without the string "lora_" in their names
 lora.mark_only_lora_as_trainable(model)
 # Training loop
 for batch in dataloader:
    ...
 ```
 4. When saving a checkpoint, generate a `state_dict` that only contains LoRA parameters.
 ```
 # ===== Before =====
 # torch.save(model.state_dict(), checkpoint_path)
 # ===== After =====
 torch.save(lora.lora_state_dict(model), checkpoint_path)
 ```
 5. When loading a checkpoint using `load_state_dict`, be sure to set `strict=False`.
 ```
 # Load the pretrained checkpoint first
 model.load_state_dict(torch.load('ckpt_pretrained.pt'), strict=False)
 # Then load the LoRA checkpoint
 model.load_state_dict(torch.load('ckpt_lora.pt'), strict=False)
 ```

#### Now training can proceed as usual.

## Additional Notes

1. While we focus on a simple yet effect setup, namely adapting only the `q` and `v` projection in a Transformer, in our examples, LoRA can be apply to any subsets of pre-trained weights. We encourage you to explore different configurations, such as adapting the embedding layer by replacing `nn.Embedding` with `lora.Embedding` and/or adapting the MLP layers. It's very likely that the optimal configuration varies for different model architectures and tasks.

2. Some Transformer implementation uses a single `nn.Linear` for the projection matrices for query, key, and value. If one wishes to constrain the rank of the updates to the individual matrices, one has to either break it up into three separate matrices or use `lora.MergedLinear`. Make sure to modify the checkpoint accordingly if you choose to break up the layer.
```
# ===== Before =====
# qkv_proj = nn.Linear(d_model, 3*d_model)
# ===== After =====
# Break it up (remember to modify the pretrained checkpoint accordingly)
q_proj = lora.Linear(d_model, d_model, r=8)
k_proj = nn.Linear(d_model, d_model)
v_proj = lora.Linear(d_model, d_model, r=8)
# Alternatively, use lora.MergedLinear (recommended)
qkv_proj = lora.MergedLinear(d_model, 3*d_model, r=8, enable_lora=[True, False, True])
```
3. Training bias vectors in tandem with LoRA might be a cost-efficient way to squeeze out extra task performance (if you tune the learning rate carefully). While we did not study its effect thoroughly in our paper, we make it easy to try in `lora`. You can mark some biases as trainable by passing "all" or "lora_only" to `bias=` when calling `mark_only_lora_as_trainable`. Remember to pass the corresponding `bias=` argument to `lora_state_dict` when saving a checkpoint.
```
# ===== Before =====
# lora.mark_only_lora_as_trainable(model) # Not training any bias vectors
# ===== After =====
# Training all bias vectors associated with modules we apply LoRA to 
lora.mark_only_lora_as_trainable(model, bias='lora_only')
# Alternatively, we can train *all* bias vectors in the model, including LayerNorm biases
lora.mark_only_lora_as_trainable(model, bias='all')
# When saving a checkpoint, use the same bias= ('all' or 'lora_only')
torch.save(lora.lora_state_dict(model, bias='all'), checkpoint_path)
```
4. Calling `model.eval()` will trigger the merging of LoRA parameters with the corresponding pretrained ones, which eliminates additional latency for subsequent forward passes. Calling `model.train()` again will undo the merge. This can be disabled by passing `merge_weights=False` to LoRA layers.

## Contact
Please contact us or post an issue if you have any questions.

For questions related to the package `loralib`:
* Edward Hu (edwardhu@microsoft.com)
* Phillip Wallis (phwallis@microsoft.com)
* Weizhu Chen (wzchen@microsoft.com)

The GPT-2 example:
* Phillip Wallis (phwallis@microsoft.com)
* Yelong Shen (yeshe@microsoft.com)

The RoBERTa/DeBERTa example:
* Lu Wang (luw@microsoft.com)

## Acknowledgements
We thank in alphabetical order Jianfeng Gao, Jade Huang, Jiayuan Huang, Lisa Xiang Li, Xiaodong Liu, Yabin Liu, Benjamin Van Durme, Luis Vargas, Haoran Wei, Peter Welinder, and Greg Yang for providing valuable feedback.

## Citation
```
@misc{hu2021lora,
    title={LoRA: Low-Rank Adaptation of Large Language Models},
    author={Hu, Edward and Shen, Yelong and Wallis, Phil and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Lu and Chen, Weizhu},
    year={2021},
    eprint={2106.09685},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
