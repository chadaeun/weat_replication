# WEAT replication

replication of Word Embedding Association Test(WEAT), which is suggested in [Semantics derived automatically from language corpora necessarily contain human biases](http://www.cs.bath.ac.uk/~jjb/ftp/CaliskanSemantics-Arxiv.pdf) by Aylin Caliskan-Islam, Joanna J. Bryson, and Arvind Narayanan.

**Note that this code is not the code used to generate results in the paper. I'm not an author of the paper, and I just replicated this code based on the paper. Thus this code can be inaccurate.**

Please let me know if there is the code provided by the authors of the paper, if there are some problems about license, or etc. I also welcome suggestions to make my codes better.

---

replicated features:

(strike-outed features are not replicated currently, but I'm planning to replicate them later)

- Word Embeddings:
    - Word2Vec
    - Glove
    - Word Embeddings availalbe at TensorFlow Hub
    - Universal Sentence Encoder
- WEAT:
    - WEAT, ~~p-value~~
    - ~~WEFAT, p-value~~
- Output Format:
    - CSV
    
---

[NumPy](http://www.numpy.org/), [Pandas](https://pandas.pydata.org/), and [tabulate](https://pypi.org/project/tabulate/) are required

[Gensim]() is required to run Word2Vec

[TensorFlow 1.7 or above](https://www.tensorflow.org/install/) and [TensorFlow hub](https://www.tensorflow.org/hub/installation) is required to run tensorflow hub

TensorFlow 1.7 is required to run Universal Sentence Encoder


## Python scripts:

### convert_weat.py

I provide target words and attributes word used in WEAT in `weat/` directory. I copied words from the paper and saved them in format presented below.

```
TARGET_LABEL_1: WORDS, WORDS, WORDS
TARGET_LABEL_2: WORDS, WORDS, WORDS
ATTRIBUTE_LABEL_1: WORDS, WORDS, WORDS
ATTRIBUTE_LABEL_2: WORDS, WORDS, WORDS
```

**There are some files which have only one target label. These files will be ignored cause WEFAT is not replicated yet.**

`conver_weat.py` collects these files and writes JSON file which inclues contents of these datas.

```
convert_weat.py [-h] --weat_dir WEAT_DIR --output OUTPUT
```

- `weat_dir`: WEAT data directory
- `output`: Output JSON file path

Format of output JSON file is presented below.

```
{
    DATA_NAME:
    {
        "method": "weat", 
        TARGET_LABEL_1: [WORDS, WORDS, WORDS],
        TARGET_LABEL_2: [WORDS, WORDS, WORDS],
        ATTRIBUTE_LABEL_1: [WORDS, WORDS, WORDS],
        ATTRIBUTE_LABEL_2: [WORDS, WORDS, WORDS],
        "X_key": TARGET_LABEL_1,
        "Y_key": TARGET_LABEL_2,
        "A_key": ATTRIBUTE_LABEL_1,
        "B_key": ATTRIBUTE_LABEL_2,
        "targets": TARGET_LABEL_1 vs TARGET_LABEL_2,
        "attributes": ATTRIBUTE_LABEL_1 vs ATTRIBUTE_LABEL_2,
    }

    DATA_NAME:
    {
        "method": "wefat", 
        TARGET_LABEL: [WORDS, WORDS, WORDS],
        ATTRIBUTE_LABEL_1: [WORDS, WORDS, WORDS],
        ATTRIBUTE_LABEL_2: [WORDS, WORDS, WORDS],
        "W_key": TARGET_LABEL,
        "A_key": ATTRIBUTE_LABEL_1,
        "B_key": ATTRIBUTE_LABEL_2,
        "targets": TARGET_LABEL,
        "attributes": ATTRIBUTE_LABEL_1 vs ATTRIBUTE_LABEL_2,
    }
}
```
### weat_test.py

`weat_test` runs WEAT on pre-trained word embeddings.

```
weat_test.py [-h] --word_embedding_type WORD_EMBEDDING_TYPE
                    [--word_embedding_path WORD_EMBEDDING_PATH]
                    [--weat_path WEAT_PATH] [--output OUTPUT]
                    [--tf_hub TF_HUB]
```

- `word_embedding_type`: Type of pretrained word embedding (`word2vec`, `glove`, or `tf_hub`)
- `word_embedding_path`: Path of pretrained word embedding (ignored when `word_embedding_type` is `tf_hub`)
- `weat_path`: Path of WEAT words file (weat.json)
- `output`: Path of output file (CSV formatted WEAT score)
- `tf_hub`: Tensorflow Hub URL (ignored when `word_embedding_type` is not `tf_hub`)

Examples of output CSV file are presented in `output/` directory.

### table_result.py

`table_result` collects results from output directory, and writes result CSV file.

```
table_result.py [-h] [--output_dir OUTPUT_DIR]
                    [--weat_path WEAT_PATH] [--result_path RESULT_PATH]
```

- `output_dir`: Directory of `weat_score.py` output files
- `weat_path`: WEAT json file path (`weat.json`)
- `result_path`: Result CSV file path

Example of result CSV file is presented in `result.csv`.

Word Embeddings Used in `result.csv`:

- [nnlm_en_dim50](https://www.tensorflow.org/hub/modules/google/nnlm-en-dim50/1): https://tfhub.dev/google/nnlm-en-dim50/1
- [nnlm_en_dim128](https://www.tensorflow.org/hub/modules/google/nnlm-en-dim128/1): https://tfhub.dev/google/nnlm-en-dim128/1
- [GloVe](https://nlp.stanford.edu/projects/glove/): Wikipedia 2014 + Gigaword 5, 50d
- [Word2Vec](https://code.google.com/archive/p/word2vec/): [GoogleNews-vectors-negative300.bin.gz](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)
- [Universal Sentence Encoder](https://www.tensorflow.org/hub/modules/google/universal-sentence-encoder/2): https://tfhub.dev/google/universal-sentence-encoder/2


## Directory

- `lib/`: Modules used in python scripts
- `output/`: Default output directory
- `weat/`: Default WEAT words directory
- `convert_weat.py`
- `weat_test.py`
- `table_result.py`
- `command.txt`: Example executing command for `weat_test.py`
- `result.csv`: Example result file

## References

- Caliskan-Islam, Aylin, Joanna J. Bryson, and Arvind Narayanan. "Semantics derived automatically from language corpora necessarily contain human biases." *arXiv preprint arXiv:1608.07187 (2016): 1-14.*