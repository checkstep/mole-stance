import argparse
import json
import logging
import os
import pathlib
from collections import defaultdict, deque
from functools import partial

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange
from transformers import AdamW, AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup

import wandb
from stancedetection.data import iterators as data_iterators
from stancedetection.data.loaders import DALoader
from stancedetection.models.domain_adaptation import MultiViewRobertaShared
from stancedetection.util.mappings import DOMAIN_TASK_MAPPINGS, TASK_MAPPINGS
from stancedetection.util.model_utils import batch_to_device, freeze_layers, get_learning_rate
from stancedetection.util.util import NpEncoder, configure_logging, set_seed

logger = logging.getLogger(__name__)

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_optimizer(model, total_steps, args):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_proportion * total_steps,
        num_training_steps=total_steps,
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        map_location = "cpu" if args.no_cuda else DEFAULT_DEVICE
        optimizer_path = os.path.join(args.model_name_or_path, "optimizer.pt")
        scheduler_path = os.path.join(args.model_name_or_path, "scheduler.pt")
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=map_location))
        scheduler.load_state_dict(torch.load(scheduler_path, map_location=map_location))
        logger.info("Loaded the saved scheduler and optimizer.")
    return optimizer, scheduler


def mean_deque(deq):
    return sum(list(deq)) / len(deq)


def calc_metrics(predictions, id2label):
    y_true = predictions["true_stance"]
    y_pred = predictions["pred_stance"]
    conf_matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(id2label)))

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


def print_metrics(metrics, is_test=False):
    set_name = "Test" if is_test else "Dev"
    for metric, value in metrics.items():
        logger.info("%s %s: %.3f", set_name, metric, value)


def evaluate_and_export(model, datasets, subset_name, args):
    # Always put is_test=True to avoid shuffling
    predictions, eval_loss = evaluate(
        model, datasets, batch_size=args.eval_batch_size, is_test=True
    )
    metrics = calc_metrics(predictions, model.config.id2label)
    metrics["loss"] = eval_loss
    print_metrics(metrics, is_test="test" == subset_name)

    for metric, value in metrics.items():
        wandb.run.summary[f"summary_{subset_name}_{metric}"] = value

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    export_metrics(metrics, args.output_dir, prefix=subset_name + "_")
    predictions_df = pd.DataFrame(predictions)
    predictions_df["pred_stance_label"] = predictions_df.pred_stance.apply(
        model.config.id2label.get
    )

    dataset_df = []
    for dataset in datasets:
        dataset_df.append(
            (
                pd.DataFrame(dataset.dataset)
                .rename({"label": "true_stance_label"}, axis=1)
                .drop(["hypothesis", "premise"], axis=1)
                .set_index("index")
            )
        )
    dataset_df = pd.concat(dataset_df, axis=0)
    assert dataset_df.shape[0] == predictions_df.shape[0]
    predictions_df = predictions_df.join(dataset_df, how="inner")
    assert dataset_df.shape[0] == predictions_df.shape[0]
    predictions_df.to_csv(os.path.join(args.output_dir, f"{subset_name}_predictions.csv"))


def print_subset_statistics(subset, id2label, subset_name):
    logger.info("----- Loaded %s %d examples. -----", subset_name, len(subset))
    label_stats = subset.label.value_counts(normalize=True)

    # Use the string labels if the task is out of domain, and not present in the labels
    label_stats.index = [i if isinstance(i, str) else id2label[i] for i in label_stats.index]
    for k, v in (label_stats * 100).round(2).items():
        logger.info("  Label:\t%s\t%.2f%%", k, v)


def task_to_global_label(x, label2id):
    task_class_key = f"{x.task_name}__{TASK_MAPPINGS[x.task_name]['id2label'][x.label]}"
    # Use the string labels if the task is out of domain, and not present in the labels
    global_label = label2id.get(task_class_key, task_class_key)

    return global_label


def create_datasets(args, tokenizer, model_config):
    datasets = defaultdict(list)
    fill_label_maps = not model_config or not hasattr(model_config, "task2id")

    if fill_label_maps:
        labelmaps = {
            "label2id": {},
            "task2id": {},
            "task2labels": [[] for _ in range(len(args.task_names))],
            "label2task": {},
            "task2domain": {},
            "domain2id": {},
        }
        for domain, tasks in DOMAIN_TASK_MAPPINGS.items():
            labelmaps["task2domain"].update({t: domain for t in tasks})
    else:
        labelmaps = {
            "id2label": model_config.id2label,
            "label2id": model_config.label2id,
            "task2id": model_config.task2id,
            "task2labels": model_config.task2labels,
            "label2task": model_config.label2task,
            "task2domain": model_config.task2domain,
            "domain2id": model_config.domain2id,
        }

    for task in args.task_names:
        task_description = TASK_MAPPINGS[task]

        task_id2label = dict(enumerate(["positive", "negative", "discuss", "other", "neutral"]))
        data_dir = pathlib.Path(args.data_dir) / task_description["task_dir"]
        dataset_loader = DALoader(data_dir, task, task_id2label)

        if args.do_train:
            datasets["train"].append(dataset_loader.train_dataset)

        if args.do_eval:
            datasets["val"].append(dataset_loader.val_dataset)
            datasets["test"].append(dataset_loader.test_dataset)

    if fill_label_maps:
        start_id = len(labelmaps["label2id"])
        task_label2id = {f"{task}__{lbl}": (start_id + id_) for id_, lbl in task_id2label.items()}
        labelmaps["label2id"].update(task_label2id)

        task_id = len(labelmaps["task2id"])
        labelmaps["task2id"][task] = task_id
        for label_id in sorted(task_label2id.values()):
            labelmaps["task2labels"][task_id].append(label_id)
            labelmaps["label2task"][label_id] = task_id

        domain_name = labelmaps["task2domain"][task]
        if domain_name not in labelmaps["domain2id"]:
            labelmaps["domain2id"][domain_name] = len(labelmaps["domain2id"])

    if fill_label_maps:
        labelmaps["id2label"] = {v: k for k, v in labelmaps["label2id"].items()}

    logger.info("  Label to ID mappings for the tasks:")
    logger.info(json.dumps(labelmaps, sort_keys=True, indent=2))

    data_iters = {"train": [], "val": [], "test": []}
    for subset_name, task_datasets in datasets.items():
        merged_datasets = defaultdict(list)

        for dataset in task_datasets:
            task_name = dataset.iloc[0].task_name
            domain_name = labelmaps["task2domain"][task_name]
            merged_datasets[domain_name].append(dataset)

        for domain_name, datasets in merged_datasets.items():
            merged_dataset: pd.DataFrame = pd.concat(datasets, axis=0)
            merged_dataset["label"] = merged_dataset.apply(
                task_to_global_label, axis=1, label2id=labelmaps["label2id"]
            )
            data_iter = data_iterators.StanceDataset(
                merged_dataset,
                tokenizer,
                args.max_seq_length,
                labelmaps["id2label"],
                labelmaps["task2labels"],
                False,
            )
            data_iter.domain_name = domain_name
            data_iters[subset_name].append(data_iter)
            print_subset_statistics(
                merged_dataset, labelmaps["id2label"], subset_name + " / " + domain_name
            )

    return data_iters, labelmaps if fill_label_maps else None


def load_model_from_pretrained(
    model_name_or_path,
    model_config,
    labelmaps=None,
    cache_dir=None,
    replace_classification=False,
):
    if labelmaps and replace_classification and "LABEL" in model_config.id2label[0]:
        model_config.num_labels = len(labelmaps["id2label"])
        model_config.label_embedding_hidden = model_config.hidden_size
        model_config.id2label = labelmaps["id2label"]
        model_config.label2id = labelmaps["label2id"]
        model_config.task2labels = labelmaps["task2labels"]
        model_config.num_domains = len(labelmaps["domain2id"])
        model_config.return_ltn_scores = False
        model_config.return_ltn_loss = False

    from_pretrained_kwargs = {
        "config": model_config,
        "from_tf": bool(".ckpt" in model_name_or_path),
        "cache_dir": cache_dir,
    }

    model = MultiViewRobertaShared.from_pretrained(model_name_or_path, **from_pretrained_kwargs)

    if replace_classification:
        logger.info("  Adding task information to the model's config...")

        for k, v in labelmaps.items():
            setattr(model.config, k, v)

    logger.info("  Number of labels is set to %d", model.num_labels)
    logger.info(model.config)

    return model.to(DEFAULT_DEVICE)


def export_metrics(metrics, output_dir, prefix=""):
    logger.info("Saving metrics to %s", output_dir)
    with open(os.path.join(output_dir, f"{prefix}metrics.json"), "w") as fp:
        json.dump(metrics, fp, sort_keys=True, cls=NpEncoder, indent=2)


def export_model(model, tokenizer, optimizer, scheduler, args, metrics, global_step):
    # Save model checkpoint
    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    logger.info("Saving model checkpoint to %s", output_dir)
    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    logger.info("Saving optimizer and scheduler states to %s", output_dir)
    export_metrics(metrics, output_dir)


def local_to_global_labels(labels, global_label_ids):
    return np.fromiter(
        (global_label_ids[pred_cid] for pred_cid in labels),
        dtype=np.long,
    )


def train(model, tokenizer, optimizer, scheduler, train_datasets, val_datasets, args):
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )

    best_metric = -np.inf
    checkpoint_stats = {}
    loss_history = deque(maxlen=10)
    acc_history = deque(maxlen=10 * args.gradient_accumulation_steps)

    model.zero_grad()
    stop_training = False

    # Adding +1 to account for missed steps due to scaling of the loss in mixed precision
    num_epochs = args.num_train_epochs + int(args.fp16 and not args.no_cuda)
    train_iterator = trange(num_epochs, position=0, leave=True, desc="Epoch")
    collate_fn = partial(data_iterators.collate_fn, add_san_masks=False, add_lel_masks=True)

    for _ in train_iterator:
        dataloaders = [
            DataLoader(
                dataset,
                shuffle=True,
                batch_size=args.train_batch_size,
                pin_memory=True,
                drop_last=False,
                collate_fn=collate_fn,
            )
            for dataset in train_datasets
        ]

        total_steps = sum(len(dl) for dl in dataloaders)
        dataloaders = [iter(dl) for dl in dataloaders]
        epoch_iterator = trange(
            total_steps,
            position=1,
            leave=True,
            desc="Iteration loss nan acc nan lr 0.0 opt. step 0",
        )
        tr_loss = 0
        for step in epoch_iterator:
            batch = None
            while not batch:
                try:
                    dataset_id = np.random.randint(0, len(train_datasets), 1)[0]
                    domain_name = train_datasets[dataset_id].domain_name
                    batch = next(dataloaders[dataset_id])
                except StopIteration:
                    pass

            batch = batch_to_device(batch, device=DEFAULT_DEVICE)
            batch["domain_name"] = domain_name

            model.train()
            loss, logits = model(**batch)

            labels = batch["labels"].detach().cpu().numpy()
            logits = logits.argmax(-1).detach().cpu().numpy()
            acc = (labels == logits).mean()

            acc_history.append(acc)

            loss = loss / args.gradient_accumulation_steps
            tr_loss += loss

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if step % args.gradient_accumulation_steps == 0:

                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()

                global_step = get_optimizer_step(optimizer)
                learning_rate = get_learning_rate(optimizer)

                loss_history.append(tr_loss)
                epoch_iterator.set_description(
                    "Iteration loss {:.4f} acc {:.4f} lr {:.2e} opt. step {}".format(
                        mean_deque(loss_history),
                        mean_deque(acc_history),
                        learning_rate,
                        global_step,
                    )
                )

                accuracy = float(np.mean(list(acc_history)[-args.gradient_accumulation_steps :]))

                wandb.log(
                    {
                        "train": {
                            "loss": tr_loss,
                            "accuracy": accuracy,
                            "learning_rate": learning_rate,
                        }
                    },
                    step=global_step,
                )

                tr_loss = 0

            global_step = get_optimizer_step(optimizer)

            if (
                args.logging_steps > 0
                and global_step % args.logging_steps == 0
                # To avoid gradient accumulation duplicates
                and global_step not in checkpoint_stats
            ):
                best_metric, metrics = evaluate_and_compare_val(
                    model, tokenizer, optimizer, scheduler, val_datasets, best_metric, args
                )
                checkpoint_stats[global_step] = metrics

            if (global_step > args.max_steps > 0) or (abs(get_learning_rate(optimizer) < 1e-12)):
                epoch_iterator.close()
                stop_training = True
                break

        if stop_training:
            train_iterator.close()
            break

    # Check the stats on the last step
    if get_optimizer_step(optimizer) not in checkpoint_stats:
        _, metrics = evaluate_and_compare_val(
            model, tokenizer, optimizer, scheduler, val_datasets, best_metric, args
        )
        checkpoint_stats[get_optimizer_step(optimizer)] = metrics

    export_metrics(checkpoint_stats, args.output_dir, "checkpoint_")
    checkpoint_stats = [
        (k, v)
        for k, v in sorted(
            checkpoint_stats.items(), key=lambda item: -item[1][args.evaluation_metric]
        )
    ]

    logger.info("Training summary:")
    logger.info(json.dumps(checkpoint_stats, indent=2, cls=NpEncoder))

    return checkpoint_stats


def evaluate_and_compare_val(
    model, tokenizer, optimizer, scheduler, val_dataset, best_metric, args
):
    metrics = evaluate_with_metrics(
        model,
        val_dataset,
        args.eval_batch_size,
        is_test=False,
    )
    print_metrics(metrics, is_test=False)

    global_step = get_optimizer_step(optimizer)
    wandb.log({"validation": metrics}, step=global_step)
    export_model(model, tokenizer, optimizer, scheduler, args, metrics, str(global_step))
    if best_metric < metrics[args.evaluation_metric]:
        logger.info(
            "Found new best model (%.2f old, %.2f new), exporting...",
            best_metric,
            metrics[args.evaluation_metric],
        )
        export_model(model, tokenizer, optimizer, scheduler, args, metrics, "best")
        best_metric = metrics[args.evaluation_metric]

    return best_metric, metrics


def evaluate_with_metrics(model, dataset, batch_size, is_test=False, num_workers=0):
    predictions, eval_loss = evaluate(
        model, dataset, batch_size=batch_size, is_test=is_test, num_workers=num_workers
    )
    metrics = calc_metrics(predictions, model.config.id2label)
    metrics["loss"] = eval_loss
    return metrics


def get_optimizer_step(optimizer):
    try:
        for params in optimizer.param_groups[0]["params"]:
            params_state = optimizer.state[params]
            if "step" in params_state:
                return params_state["step"]

        return -1
    except KeyError:
        return -1


@torch.no_grad()
def evaluate(
    model,
    datasets,
    batch_size,
    is_test=False,
    num_workers=0,
    max_steps=None,
):
    collate_fn = partial(data_iterators.collate_fn, add_san_masks=False, add_lel_masks=True)
    dataloaders = [
        DataLoader(
            dataset,
            shuffle=not is_test,
            batch_size=batch_size,
            pin_memory=True,
            drop_last=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
        for dataset in datasets
    ]

    total_steps = sum(len(dataloader) for dataloader in dataloaders)

    predictions = defaultdict(list)
    model.eval()
    eval_loss = 0
    last_step = 0
    with tqdm(
        position=0 if is_test else 2, leave=True, desc="Evaluating", total=total_steps
    ) as data_iter:
        for _, dataloader in enumerate(dataloaders):
            for batch in dataloader:
                if max_steps and last_step > max_steps:
                    break

                batch = batch_to_device(batch, device=DEFAULT_DEVICE)

                loss, logits = model(**batch)
                eval_loss += loss.item()

                probs = logits.softmax(-1).detach().cpu().numpy()
                predictions["probs"] += probs.tolist()
                predictions["pred_stance"] += probs.argmax(-1).tolist()
                predictions["true_stance"] += batch["labels"].detach().cpu().tolist()

                last_step = last_step + 1
                data_iter.update(1)

    return predictions, eval_loss / last_step


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--task_names",
        type=str,
        required=True,
        nargs="+",
        choices=TASK_MAPPINGS.keys(),
        help="The name of the task to train selected in the list: "
        + ", ".join(TASK_MAPPINGS.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--evaluation_metric",
        type=str,
        default="f1_macro",
        choices=["f1_macro", "precision_macro", "recall_macro", "accuracy"],
        help="This metric is used to select the best model from the dev set.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )

    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the test set."
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Rul evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0,
        type=float,
        help="Linear warmup over warmup_steps.",
    )
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument(
        "--log_on_epoch",
        action="store_true",
        help="Log every epoch. Overwrites the logging_steps parameter.",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--replace_classification",
        action="store_true",
        help="Replace the classification layer",
    )
    parser.add_argument(
        "--domain_adversarial",
        action="store_true",
        help="Use domain adversarial loss",
    )
    parser.add_argument(
        "--freeze_embeddings",
        action="store_true",
        help="Whether to freeze the embedding layer.",
    )
    parser.add_argument(
        "--freeze_layers", nargs="*", help="Whether to freeze layers with given ids.", type=int
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    args = parser.parse_args()

    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Remove duplicates and sort alphabetically
    args.task_names = sorted(list(set(args.task_names)))

    if args.no_cuda:
        global DEFAULT_DEVICE
        DEFAULT_DEVICE = "cpu"

    set_seed(args.seed)
    configure_logging()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir,
        use_fast=False,
    )

    model_config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir,
    )

    model_config.is_domain_adversarial = args.domain_adversarial

    logger.info("***** Loading the dataset *****")

    data_iters, labelmaps = create_datasets(
        args, tokenizer, model_config if not args.replace_classification else None
    )

    model = load_model_from_pretrained(
        args.model_name_or_path,
        model_config,
        labelmaps=labelmaps,
        cache_dir=args.cache_dir,
        replace_classification=args.replace_classification,
    )
    freeze_layers(model.roberta, args.freeze_embeddings, args.freeze_layers)

    logger.info("Training/evaluation parameters %s", args)
    wandb.login()

    wandb.init(config=args)

    if args.do_train:
        wandb.watch(model)
        total_examples = sum(len(dataset) for dataset in data_iters["train"])
        steps_per_epoch = (
            int(np.ceil(total_examples / args.train_batch_size)) // args.gradient_accumulation_steps
        )
        total_steps = steps_per_epoch * args.num_train_epochs
        optimizer, scheduler = build_optimizer(model, total_steps, args)

        if args.log_on_epoch:
            args.logging_steps = steps_per_epoch

        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
                )
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(data_iters["train"]))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            args.train_batch_size * args.gradient_accumulation_steps,
        )
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", total_steps)
        logger.info("  Logging steps = %d ", args.logging_steps)

        checkpoint_stats = train(
            model,
            tokenizer,
            optimizer,
            scheduler,
            data_iters["train"],
            data_iters["val"],
            args,
        )

        if checkpoint_stats:
            model_id = "best"
            best_model_path = os.path.join(args.output_dir, "checkpoint-{}".format(model_id))
            logger.info("Loading best model on validation from %s", best_model_path)

            model_config = AutoConfig.from_pretrained(
                best_model_path,
                cache_dir=args.cache_dir,
            )

            model = load_model_from_pretrained(
                best_model_path,
                model_config,
                cache_dir=args.cache_dir,
                replace_classification=False,
            )

    if args.do_eval:
        evaluate_and_export(model, data_iters["val"], "val", args)
        evaluate_and_export(model, data_iters["test"], "test", args)


if __name__ == "__main__":
    main()
