import argparse
import json
import logging
import pathlib
import pickle
import warnings
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm
from transformers import AutoConfig

from stancedetection.models.trainer import TASK_MAPPINGS
from stancedetection.util.util import NpEncoder

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

logger = logging.getLogger(__name__)
tqdm.pandas()
pd.set_option("display.max_rows", 500)


def configure_logging():
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.ERROR,
    )


def load_glove(vectors_path: pathlib.Path):
    if vectors_path.suffix == ".bin":
        with vectors_path.open("rb") as fp:
            vectors = pickle.load(fp)
        return vectors

    vectors = {}
    with vectors_path.open("r") as fp:
        for line in tqdm(fp):
            glove = line.split(" ")
            word = glove[0]
            vectors[word] = np.fromiter((float(x) for x in glove[1:]), dtype=np.float32)

    with vectors_path.with_suffix(".bin").open("wb") as fp:
        pickle.dump(vectors, fp)

    return vectors


def cos_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def label_to_vector(label, vectors):
    task, label = label.split("__")
    label_words = label.lower().replace("_", " ")
    label_vec = vectors[label_words]

    return label_vec


def map_label_to_possibles(
    row: pd.Series,
    vectors: Dict[str, Any],
    id2label,
    possible_classes: List[str],
    cosine_rerank=True,
):
    if not cosine_rerank:
        label = row.pred_stance_label
        # This won't work for OOV words
        label_vec = label_to_vector(label, vectors)

        winner = None
        max_cos = -np.inf
        for class_ in possible_classes:
            class_vec = label_to_vector("__" + class_.lower(), vectors)
            cos = cos_similarity(class_vec, label_vec)
            if cos > max_cos:
                max_cos = cos
                winner = class_
    else:
        probs = np.array(row.probs, dtype=np.float32)
        updated_probs = np.tile(probs, (len(possible_classes), 1))
        for id_, label in id2label.items():
            label_vec = label_to_vector(label, vectors)
            for j, class_ in enumerate(possible_classes):
                class_vec = label_to_vector("__" + class_.lower(), vectors)
                updated_probs[j, id_] *= (cos_similarity(label_vec, class_vec) + 1) / 2

        winner = possible_classes[updated_probs.max(-1).argmax(-1)]

    return winner


def calc_metrics(predictions):
    y_true = predictions["true_stance_label"]
    y_pred = predictions["pred_possible_label"]

    id2label = {i: l for i, l in enumerate(sorted(set(y_true.tolist() + y_pred.tolist())))}
    conf_matrix = confusion_matrix(y_true, y_pred)

    metrics = {}
    diag_values = conf_matrix[np.diag_indices(len(conf_matrix))]
    metrics["accuracy"] = diag_values.sum() / conf_matrix.sum()

    # Per class metrics
    clw_precision = np.nan_to_num(diag_values / conf_matrix.sum(axis=0), nan=0.0)
    clw_recall = np.nan_to_num(diag_values / conf_matrix.sum(axis=1), nan=0.0)
    clw_f1 = 2 * clw_precision * clw_recall / (clw_precision + clw_recall)
    clw_f1 = np.nan_to_num(clw_f1, nan=0.0)

    # Macro average
    metrics["precision_macro"] = clw_precision.mean()
    metrics["recall_macro"] = clw_recall.mean()
    metrics["f1_macro"] = clw_f1.mean()

    per_task_metrics = defaultdict(list)

    for i, true in id2label.items():
        metrics[f"precision_clw_{true}"] = clw_precision[i]
        metrics[f"recall_clw_{true}"] = clw_recall[i]
        metrics[f"f1_clw_{true}"] = clw_f1[i]

        task = true.split("__")[0]
        per_task_metrics[f"precision_task_{task}"].append(clw_precision[i])
        per_task_metrics[f"recall_task_{task}"].append(clw_recall[i])
        per_task_metrics[f"f1_task_{task}"].append(clw_f1[i])

    metrics.update({k: np.mean(v) for k, v in per_task_metrics.items()})
    metrics = json.loads(json.dumps(metrics, sort_keys=True, cls=NpEncoder))

    return metrics


def evaluate_ood(
    predictions: pd.DataFrame,
    vectors: Dict[str, Any],
    id2label: Dict[int, str],
    possible_classes: List[str],
):
    predictions["pred_stance_label"] = predictions.pred_stance.apply(id2label.get)
    predictions["pred_possible_label"] = predictions.apply(
        map_label_to_possibles,
        axis=1,
        vectors=vectors,
        id2label=id2label,
        possible_classes=possible_classes,
        cosine_rerank=False,
    )
    predictions["pred_possible_label"] = (
        predictions["task_name"] + "__" + predictions["pred_possible_label"]
    )
    logger.info("\n%s", json.dumps(calc_metrics(predictions), indent=2, sort_keys=True))
    return calc_metrics(predictions)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--predictions_path",
        default=None,
        type=pathlib.Path,
        required=True,
        help="The input predictions file.",
    )
    parser.add_argument(
        "--embeddings_path",
        default=None,
        type=pathlib.Path,
        required=True,
        help="The embeddings vectors used for similarity.",
    )
    parser.add_argument(
        "--embeddings",
        default="glove",
        type=str,
        required=True,
        help="Embeddings type.",
    )
    parser.add_argument(
        "--model_config_path",
        default=None,
        type=pathlib.Path,
        required=True,
        help="The model's config.",
    )
    parser.add_argument(
        "--target_task",
        default=None,
        type=str,
        required=True,
        choices=TASK_MAPPINGS.keys(),
        help="Target class to load target labels for.",
    )

    args = parser.parse_args()
    configure_logging()

    possible_classes = list(TASK_MAPPINGS[args.target_task]["id2label"].values())
    logger.info("Possible classes: %s", ",".join(possible_classes))

    id2label = AutoConfig.from_pretrained(args.model_config_path).id2label
    logger.info("Loaded id2labels from the model's config.")
    logger.info("%s", json.dumps(id2label, sort_keys=True))

    vectors = pd.read_parquet(args.embeddings_path)[args.embeddings].to_dict()
    logger.info("Loaded %d %s vectors.", len(vectors), args.embeddings)

    label_vectors = {}
    for _, label in id2label.items():
        label_vectors[label] = label_to_vector(label, vectors).tolist()

    predictions = pd.read_csv(
        args.predictions_path,
        index_col=0,
        header=0,
        sep=",",
        converters={"probs": lambda x: eval(x)},
    )
    logger.info("Loaded %d predictions.", len(predictions))

    metrics = evaluate_ood(predictions, vectors, id2label, possible_classes)
    ood_metrics_path = (
        args.predictions_path.parent
        / args.embeddings
        / ("_".join(("ood", args.predictions_path.stem, "metrics")) + ".json")
    )
    ood_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with ood_metrics_path.open("w") as fp:
        json.dump(metrics, fp, indent=2, sort_keys=True)
        logger.info("Dumped out-of-domain metrics into %s.", str(ood_metrics_path))


if __name__ == "__main__":
    main()
