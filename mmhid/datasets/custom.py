from torch.utils.data import Dataset

from .builder import HID_DATASETS


@HID_DATASETS.register_module()
class CustomDataset(Dataset):
    """Custom dataset for human interaction detection.
    """
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
