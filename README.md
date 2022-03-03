# Cross-Domain Label-Adaptive Stance Detection

Stance detection concerns the classification of a writer’s viewpoint towards a target. There are different task variants, e.g., stance of a tweet vs. a full article, or stance with respect to a claim vs. an (implicit) topic. Moreover, task definitions vary, which includes the label inventory, the data collection, and the annotation protocol. All these aspects hinder cross-domain studies, as they require changes to standard domain adaptation approaches. In this paper, we perform an in-depth analysis of 16 stance detection datasets, and we explore the possibility for cross-domain learning from them. Moreover, we propose an end-to-end unsupervised framework for out-of-domain prediction of unseen, user-defined labels. In particular, we combine domain adaptation techniques such as mixture of experts and domain-adversarial training with label embeddings, and we demonstrate sizable performance gains over strong baselines, both (i) in-domain, i.e., for seen targets, and (ii) out-of-domain, i.e., for unseen targets. Finally, we perform an exhaustive analysis of the cross-domain results, and we highlight the important factors influencing the model performance.

## Setup

```console
$ python3 -m venv ~/.virtualenvs/stance-detection
$ source ~/.virtualenvs/stance-detection/bin/activate
```

### Updating project dependencies

```console
# And to install the packages
$ pip install -r requirements.txt
```

## Getting the datasets
* Stance Detection Benchmark [Data](https://github.com/UKPLab/mdl-stance-robustness#preprocessing) (arc argmin fnc1 iac1 ibmcs perspectrum scd semeval2016t6 semeval2019t7 snopes)
* Will-They-Won't-They [Data](https://github.com/cambridge-wtwt/acl2020-wtwt-tweets) (wtwt)
* Emergent [Data](https://www.dropbox.com/sh/9t7fd7xfahb0e1v/AABHcvt9dSH6RNFpnSoYqlZra/emergent?) (emergent)
* Rumor has it [Data](https://github.com/vahedq/rumors/tree/master/data) (rumor)
* Multi-Target Stance Dataset [Data](http://www.site.uottawa.ca/~diana/resources/stance_data/) (mtsd)
* Political Debates [Data](http://mpqa.cs.pitt.edu/corpora/political_debates/) (poldeb)
* VAried Stance Topics [Data](https://github.com/emilyallaway/zero-shot-stance) (vast)

### Format

The data files must be converted into json lines format:
```json
{"uid": 1234, "label": 0, "hypothesis": "TEXT OF THE CONTEXT", "premise": "TEXT OF THE TARGET"}
```

The files for each dataset must be named with the following pattern: `rumor_train.json`, `rumor_dev.json`, `rumor_test.json`, where `rumor` should be replaced with the proper name of the dataset (see DATASETS below).    

## Running the code

### Environment

```shell
DATASETS=(arc argmin fnc1 iac1 ibmcs perspectrum scd semeval2016t6 semeval2019t7 snopes emergent mtsd poldeb rumor vast wtwt)
TARGET=arc
datasets=("${DATASETS[@]/$TARGET}")

MODEL_NAME="roberta-base"
MAX_SEQ_LEN=100
LEARNING_RATE=1e-05
```

### Label Embeddings Training

```shell

python src/stancedetection/models/trainer.py --data_dir "data/all/" \
        --model_name_or_path ${MODEL_NAME} \
        --output_dir ${OUTPUT_DIR} \
        --task_names ${datasets[@]} \
        --do_train \
        --do_eval \
        --model_type lel \
        --learning_rate ${LEARNING_RATE} \
        --weight_decay 0.01 \
        --per_gpu_train_batch_size 64 \
        --per_gpu_eval_batch_size 256 \
        --replace_classification \
        --num_train_epochs ${EPOCHS} \
        --warmup_proportion ${WARMUP} \
        --adam_epsilon 1e-08 \
        --log_on_epoch \
        --max_seq_length ${MAX_SEQ_LEN} \
        --evaluate_during_training \
        --gradient_accumulation_steps 1 \
        --seed ${SEED} \
        --fp16 \
        --cache_dir cache \
        --overwrite_output_dir

python src/stancedetection/models/trainer.py --data_dir "data/all/" \
        --model_name_or_path ${OUTPUT_DIR}/checkpoint-best \
        --output_dir ${EVAL_DIR} \
        --task_names ${TARGET} \
        --do_eval \
        --model_type lel \
        --per_gpu_eval_batch_size 256 \
        --max_seq_length ${MAX_SEQ_LEN} \
        --seed ${SEED} \
        --fp16 \
        --cache_dir cache \
        --overwrite_output_dir
```

### Domain Adaptation

```shell
python src/stancedetection/models/trainer_da.py --data_dir "data/all/" \
        --model_name_or_path ${MODEL_NAME} \
        --output_dir ${OUTPUT_DIR} \
        --task_names ${datasets[@]} \
        --do_train \
        --do_eval \
        --learning_rate ${LEARNING_RATE} \
        --weight_decay 0.01 \
        --per_gpu_train_batch_size 64 \
        --per_gpu_eval_batch_size 256 \
        --replace_classification \
        --num_train_epochs ${EPOCHS} \
        --warmup_proportion ${WARMUP} \
        --adam_epsilon 1e-08 \
        --log_on_epoch \
        --max_seq_length ${MAX_SEQ_LEN} \
        --evaluate_during_training \
        --gradient_accumulation_steps 1 \
        --seed ${SEED} \
        --fp16 \
        --cache_dir cache \
        --overwrite_output_dir

python src/stancedetection/models/trainer_da.py --data_dir "data/all/" \
        --model_name_or_path ${OUTPUT_DIR}/checkpoint-best \
        --output_dir ${EVAL_DIR} \
        --task_names ${TARGET} \
        --do_eval \
        --per_gpu_eval_batch_size 256 \
        --max_seq_length ${MAX_SEQ_LEN} \
        --seed ${SEED} \
        --fp16 \
        --cache_dir cache \
        --overwrite_output_dir
```

## References

Please cite as [1]. There is also an [arXiv version](https://arxiv.org/abs/2104.07467).


[1] Momchil Hardalov, Arnav Arora, Preslav Nakov, and Isabelle Augenstein. 2021. [Cross-Domain Label-Adaptive Stance Detection](https://aclanthology.org/2021.emnlp-main.710/). In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 9011–9028, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.


```
@inproceedings{hardalov-etal-2021-cross,
    title = "Cross-Domain Label-Adaptive Stance Detection",
    author = "Hardalov, Momchil  and
      Arora, Arnav  and
      Nakov, Preslav  and
      Augenstein, Isabelle",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.710",
    doi = "10.18653/v1/2021.emnlp-main.710",
    pages = "9011--9028"
}

```


## License

The code in this repository is licenced under the [CC-BY-NC-SA 4.0](LICENSE). The datasets are licensed under [CC-BY-SA 4.0](LICENSE.data).
