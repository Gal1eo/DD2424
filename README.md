# This is an implementation of data augmentation using BERT

This repository contains MLM fine-tuning task and contextual data regeneration examples using the implementation of BERT from:

- [PyTorch Pretrained BERT](https://github.com/huggingface/pytorch-pretrained-BERT)

The training dataset comes from:

- [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)

Disclaimer: The dataset for this competition contains text that may be considered profane, vulgar, or offensive.

Thanks to "Conditional BERT Contextual Augmentation" https://arxiv.org/pdf/1812.06705.pdf for providing such a wonderful idea.

## Content

| Section | Description |
|-|-|
| [Requirements](#Requirements) | How to install the required package |
| [Usage](#Usage) | Quickstart examples |
| [Modification](#Modification) | How to train data from other sources |
| [Effect](#Effect) | The effect of augmentation using Kaggle Baseline model |
| [GPU](#GPU) | GPU requirement and memory |

## Requirements

This repo was tested on Python 3.6 and PyTorch 1.1

### Installation

PyTorch pretrained bert can be installed by pip as follows:
```bash
pip install pytorch-pretrained-bert
```
PyTorch can be installed by conda as follows:
```bash
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
```

## Usage

If you want to reproduce the results of toxic comment augmentation, you can run the command:
```bash
python finetune.py
python train_aug.py
```

## Modification

If you want to use this repo for other text data augmentation, there are some tips of modification:

- Add your own taskname under the ojcet of AugProcessor 
- If the format of your data is not .csv file, you need to modify the reading method under the object of DataProcessor

And you can run the test with the command:
```shell
python finetune.py \
  --data_dir $INPUT_DIR/$TASK_NAME \
  --output_dir $OUTPUT_DIR/$TASK_NAME \
  --task_name $TASK_NAME \
  --bert_model bert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 5e-5 \
  --num_train_epochs 10.0 \
```

```shell
python train_aug.py \
  --data_dir $INPUT_DIR/$TASK_NAME \
  --output_dir $OUTPUT_DIR/$TASK_NAME \
  --task_name $TASK_NAME \
  --bert_model bert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 5e-5 \
  --num_train_epochs 10.0 \
```

## Effect
The final result is tested on the [Kaggle Baseline model](https://www.kaggle.com/gali1eo/benchmark-kernel)

## GPU

If you want to reproduce our results with the defult settings, you need a GPU with more than 14GB memory. Otherwise you need to decrease the number of batch_size and max_seq_length.

