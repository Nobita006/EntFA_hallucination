# Inspecting the Factuality of Hallucinations in Abstractive Summarization

This directory contains code necessary to replicate the training and evaluation for the ACL 2022 paper ["Hallucinated but Factual! Inspecting the Factuality of Hallucinations in Abstractive Summarization"](https://arxiv.org/pdf/2109.09784.pdf) by [Meng Cao](https://mcao516.github.io/), [Yue Dong](https://www.cs.mcgill.ca/~ydong26/) and [Jackie Chi Kit Cheung](https://www.cs.mcgill.ca/~jcheung/).

## Dependencies and Setup
The code is based on Huggingface's [Transformers](https://github.com/huggingface/transformers) library. 
  ```
  git clone https://github.com/mcao516/EntFA.git
  cd ./EntFA

  py --list
  py -3.10 -m venv env
  .\env\Scripts\Activate
  python --version

  pip install -r requirements.txt
  pip install fairseq==0.10
  python setup.py install
  ```

## How to Run
Conditional masked language model (CMLM) checkpoint can be found [here](https://drive.google.com/drive/folders/10ibVc5R7q4Gc0TH1AIRo7IaLCV83SkpF?usp=sharing). For masked language model (MLM), download `bart.large` at Fairseq's [BART](https://github.com/pytorch/fairseq/tree/main/examples/bart) repository. Download CMLM and MLM, put them in the models directory.

### Train KNN Classifier

For Linux

```bash
OUTPUT_DIR=knn_checkpoint
mkdir $OUTPUT_DIR

python examples/train_knn.py \
  --train-path data/train.json \
  --test-path data/test.json \
  --cmlm-model-path models \
  --data-name-or-path models/xsum-bin \
  --mlm-path models/bart.large \
  --output-dir $OUTPUT_DIR;
```
You can also find an example at `examples/train_knn_classifier.ipynb`.

For windows

```
mkdir knn_checkpoint

pip install scikit-learn
pip install numpy==1.23.5
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

$env:PYTHONPATH = "C:\Users\sayan\OneDrive\Documents\Visual_Studio_2022\Priyanka\EntFA\src"

python examples/train_knn.py `
  --train_path data/train.json `
  --test_path data/test.json `
  --cmlm_model_path models `
  --data_name_or_path models/xsum-bin `
  --mlm_path models/bart.large `
  --output_dir knn_checkpoint

```

### Evaluation
Evalute the entity-level factuality of generated summaries. Input file format: one document/summary per line.

For Linux

```bash
SOURCE_PATH=test.source
TARGET_PATH=test.hypothesis

python examples/evaluation.py \
    --source-path $SOURCE_PATH \
    --target-path $TARGET_PATH \
    --cmlm-model-path models \
    --data-name-or-path models/xsum-bin \
    --mlm-path models/bart.large \
    --knn-model-path models/knn_classifier.pkl;
```

For Windows

```
python -m spacy download en_core_web_sm
python .\getTestFromCsv.py

python examples/evaluation.py `
    --source_path data/test.source `
    --target_path data/test.hypothesis `
    --cmlm_model_path models `
    --data_name_or_path models/xsum-bin `
    --mlm_path models/bart.large `
    --knn_model_path knn_checkpoint/knn_classifier.pkl
```
Also check `examples/evaluation.ipynb`.
