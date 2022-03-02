import pathlib
from abc import abstractmethod
from typing import Tuple

import pandas as pd


class BaseLoader:
    def __init__(self, dataset_path, task_name, id2label):
        self.dataset_path = pathlib.Path(dataset_path)
        self.task_name = task_name
        self.id2label = id2label
        self.label2id = {v: k for k, v in id2label.items()}
        self.train_dataset, self.val_dataset, self.test_dataset = self._prepare_splits(
            self.dataset_path, self.task_name
        )

        for subset in (self.train_dataset, self.val_dataset, self.test_dataset):
            subset["task_name"] = task_name
            if "hypothesis" not in subset.columns:
                subset["hypothesis"] = ""

            if task_name == "wtwt":
                premise_maps = {
                    "AET_HUM": "Aetna acquires Humana",
                    "ANTM_CI": "Anthem acquires Cigna",
                    "CVS_AET": "CVS Health acquires Aetna",
                    "CI_ESRX": "Cigna acquires Express Scripts",
                    "FOXA_DIS": "Disney acquires 21st Century Fox ",
                }
                subset["premise"] = subset["premise"].apply(premise_maps.get)

        self.train_dataset = self.train_dataset[self.train_dataset.hypothesis != "[not found]"]
        self.val_dataset = self.val_dataset[self.val_dataset.hypothesis != "[not found]"]
        self.test_dataset = self.test_dataset[self.test_dataset.hypothesis != "[not found]"]

        self.train_dataset = self.train_dataset.reset_index(drop=True)
        self.val_dataset = self.val_dataset.reset_index(drop=True)
        self.test_dataset = self.test_dataset.reset_index(drop=True)

    @abstractmethod
    def _prepare_splits(
        self, dataset_path: pathlib.Path, task_name: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        pass


class StanceLoader(BaseLoader):
    def _prepare_splits(self, dataset_path, task_name):
        train = pd.read_json(dataset_path / f"{task_name}_train.json", lines=True).copy()
        val = pd.read_json(dataset_path / f"{task_name}_dev.json", lines=True).copy()
        test = pd.read_json(dataset_path / f"{task_name}_test.json", lines=True).copy()

        return train, val, test


class DALoader(StanceLoader):
    def __init__(self, dataset_path, task_name, id2label):
        super().__init__(dataset_path, task_name, id2label)
        from stancedetection.util import mappings

        g2id = {k: v for v, k in enumerate(["positive", "negative", "discuss", "other", "neutral"])}
        l2g = {k: DALoader.map_to_group(v) for k, v in mappings.RELATED_TASK_MAP.items()}
        self.train_dataset.label = (
            self.train_dataset.label.apply(mappings.TASK_MAPPINGS[task_name]["id2label"].get)
            .apply(lambda x: f"{task_name}__{x}")
            .apply(lambda x: l2g[x])
            .apply(lambda x: g2id[x])
        )

        self.val_dataset = self.val_dataset.reset_index(drop=True)
        self.test_dataset = self.test_dataset.reset_index(drop=True)

    @staticmethod
    def map_to_group(label_list):
        from stancedetection.util import mappings

        if label_list is mappings.POSITIVE_LABELS:
            return "positive"
        elif label_list is mappings.NEGATIVE_LABELS:
            return "negative"
        elif label_list is mappings.DISCUSS_LABELS:
            return "discuss"
        elif label_list is mappings.OTHER_LABELS:
            return "other"
        elif label_list is mappings.NEUTRAL_LABELS:
            return "neutral"

        raise Exception(f"No such mapping for {label_list}")
