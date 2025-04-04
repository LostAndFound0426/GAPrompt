# GAPrompt

Code for the paper GAPrompt: A Soft Prompt Compression Model for Few-shot Relation Extraction

# Requirements

**Step1** Create a virtual environment using `Anaconda` and enter it.<br>


```bash
conda create -n gaprompt python=3.9

conda activate gaprompt
```
   
**Step2** install requirements:

```
pip install -r requirements.txt
```
# Datasets

You can download the dataset from this [repository](https://github.com/zjunlp/KnowPrompt/tree/master/dataset).


# How to run

## Initialize the answer words

Use the comand below to get the answer words to use in the training.

```shell
python get_label_word.py --model_name_or_path roberta-large  --dataset_name semeval
```

The `{answer_words}.pt`will be saved in the dataset, you need to assign the `model_name_or_path` and `dataset_name` in the `get_label_word.py`.

## Split few-shot dataset

Download the data first, and put it to `dataset` folder. Run the comand below, and get the few shot dataset.

```shell
python generate_k_shot.py --data_dir ./dataset --k 5 --dataset semeval
cd dataset
cd semeval
cp rel2id.json val.txt test.txt ./k-shot/5-1
```
You need to modify the `k` and `dataset` to assign k-shot and dataset. Here we default seed as 1,2,3,4,5 to split each k-shot, you can revise it in the `generate_k_shot.py`

## Run
```bash
bash scripts/semeval.sh 
bash scripts/tacred.sh
bash scripts/tacrev.sh
```

# Acknowledgement

Part of our code is borrowed from code of [RetrievalRE](https://github.com/zjunlp/PromptKG/tree/main/research/RetrievalRE). We sincerely appreciate their contribution.

