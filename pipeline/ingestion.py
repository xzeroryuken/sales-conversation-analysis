from abc import ABC, abstractmethod
import pandas as pd
from datasets import load_dataset


class DataSource(ABC):
    @abstractmethod
    def load(self) -> pd.DataFrame: ...


class HuggingFaceSource(DataSource):
    def __init__(self, dataset_name: str, split: str = "train"):
        self.dataset_name = dataset_name
        self.split = split

    def load(self) -> pd.DataFrame:
        ds = load_dataset(self.dataset_name)
        df = pd.DataFrame(ds[self.split])
        df["conversation"] = df.apply(
            lambda x: "\n".join(value for value in x if not pd.isnull(value)), axis=1
        )
        return df[["conversation"]]


class CSVSource(DataSource):
    def __init__(self, path: str, conversation_column: str = "conversation"):
        self.path = path
        self.conversation_column = conversation_column

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.path)
        return df[[self.conversation_column]].rename(
            columns={self.conversation_column: "conversation"}
        )
