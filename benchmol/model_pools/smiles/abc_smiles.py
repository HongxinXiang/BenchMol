import abc
import dataclasses


@dataclasses.dataclass
class SmilesABC(abc.ABC):

    @abc.abstractmethod
    def from_pretrained(self, pretrain_path, model_key, consistency, logger):
        pass

    @abc.abstractmethod
    def get_dataset(self, csv_path, task_type):
        pass

    @abc.abstractmethod
    def collate(self, batch):
        pass