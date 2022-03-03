from abc import abstractmethod
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

from stancedetection.util.mappings import RELATED_TASK_MAP


class BaseDataset(Dataset):
    def __init__(self, dataset, id2label, task2labels):
        super(BaseDataset, self).__init__()
        self.dataset = dataset
        self.id2label = id2label
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.task2labels = task2labels
        self.label2task = {}
        for task, labels in enumerate(task2labels):
            self.label2task.update({label: task for label in labels})

    def __len__(self):
        return len(self.dataset)

    @abstractmethod
    def __getitem__(self, idx):
        pass


class StanceDataset(BaseDataset):
    def __init__(self, dataset, tokenizer, max_seq_length, id2label, task2labels, soft_labels=True):
        super(StanceDataset, self).__init__(dataset, id2label, task2labels)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.dataset = dataset[["uid", "hypothesis", "premise", "task_name", "label"]].to_records()
        self.soft_labels = soft_labels

    def __getitem__(self, idx):
        example = self.dataset[idx]
        label = example["label"]

        if example["hypothesis"]:
            encoded_input = self.tokenizer.encode_plus(
                example["hypothesis"],
                example["premise"],
                max_length=self.max_seq_length,
                truncation="longest_first",
                return_tensors="np",
            )
            # We can't always relay that there will be token_type_ids, so we locate the first [SEP]
            h_length = (
                np.where(encoded_input["input_ids"][0] == self.tokenizer.sep_token_id)[0][0] - 1
            )
        else:
            encoded_input = self.tokenizer.encode_plus(
                example["premise"],
                max_length=self.max_seq_length,
                return_tensors="np",
                truncation=True,
            )
            h_length = 0

        # If the labels are strings, then you are in an out of domain setting, so you do not have labels

        if not isinstance(label, str):
            encoded_input["lel_mask"] = np.zeros(len(self.id2label), dtype=bool)
            if self.soft_labels:
                label_name = self.id2label[label]
                related_tasks = RELATED_TASK_MAP[label_name]
                related_tasks_ids = [
                    self.label2id[ln]
                    for ln in related_tasks
                    if ln != label_name and ln in self.label2id
                ]

                encoded_input["soft_labels"] = np.zeros(len(self.id2label), dtype=np.float)
                num_related_tasks = len(related_tasks_ids)
                per_task_portion = 0.1 / max(num_related_tasks, 1)
                encoded_input["soft_labels"][related_tasks_ids] = per_task_portion
                encoded_input["soft_labels"][label] = 1 - per_task_portion * num_related_tasks
                visible_tasks = list(
                    sorted(set(self.task2labels[self.label2task[label]]) | set(related_tasks_ids))
                )
            else:
                visible_tasks = self.task2labels[self.label2task[label]]

            encoded_input["labels"] = np.long(label)
            encoded_input["lel_mask"][visible_tasks] = True
        else:
            # You do need lel_mask, since you want to predict all labels
            encoded_input["lel_mask"] = np.ones(len(self.id2label), dtype=bool)
            encoded_input["labels"] = np.long(0)

        encoded_input["length"] = len(encoded_input["input_ids"][0])
        encoded_input["h_length"] = h_length

        return encoded_input


class StanceDatasetMTL(BaseDataset):
    def __init__(self, dataset, tokenizer, max_seq_length, id2label, task2labels):
        super(StanceDatasetMTL, self).__init__(dataset, id2label, task2labels)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.dataset = dataset[["uid", "hypothesis", "premise", "task_name", "label"]].to_records()
        self.task_name = self.dataset[0]["task_name"]

    def __getitem__(self, idx):
        example = self.dataset[idx]
        label = example["label"]

        if example["hypothesis"]:
            encoded_input = self.tokenizer.encode_plus(
                example["hypothesis"],
                example["premise"],
                max_length=self.max_seq_length,
                truncation="longest_first",
                return_tensors="np",
            )
        else:
            encoded_input = self.tokenizer.encode_plus(
                example["premise"],
                max_length=self.max_seq_length,
                return_tensors="np",
                truncation=True,
            )

        # If the labels are strings, then you are in an out of domain setting, so you do not have labels

        encoded_input["labels"] = np.long(label)
        encoded_input["length"] = len(encoded_input["input_ids"][0])

        return encoded_input


def pad_vector(a, max_len):
    return np.pad(a, (0, max_len - len(a)), mode="constant", constant_values=0)


def collate_fn(raw_batch, add_san_masks=False, add_lel_masks=False):
    max_len = max(x["length"] for x in raw_batch)

    batch = defaultdict(list)
    for row in raw_batch:
        for field in ["input_ids", "token_type_ids", "attention_mask", "labels", "soft_labels"]:
            if field not in row:
                continue

            value = row[field]
            if field != "soft_labels" and isinstance(value, np.ndarray):
                value = pad_vector(value[0], max_len)
            batch[field].append(value)

        if add_san_masks:
            premise_mask = np.ones(max_len, dtype=np.long)
            hypothesis_mask = np.ones(max_len, dtype=np.long)

            premise_mask[row["h_length"] : row["length"]] = 0
            hypothesis_mask[: row["h_length"]] = 0
            assert premise_mask.sum() >= 1
            assert hypothesis_mask.sum() > 1

            batch["premise_masks"].append(premise_mask)
            batch["hypothesis_masks"].append(hypothesis_mask)

        if add_lel_masks:
            batch["lel_mask"].append(row["lel_mask"])

    tensor_batch = {}

    for field in ["input_ids", "token_type_ids", "attention_mask", "labels", "soft_labels"]:
        if field not in batch:
            continue
        if field == "soft_labels":
            tensor_batch[field] = torch.FloatTensor(batch[field])
        else:
            tensor_batch[field] = torch.LongTensor(batch[field])

    for field in ["premise_masks", "hypothesis_masks", "lel_mask"]:
        if field not in batch:
            continue
        tensor_batch[field] = torch.ByteTensor(batch[field])

    return tensor_batch
