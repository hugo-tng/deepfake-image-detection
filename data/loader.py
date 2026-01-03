from torch.utils.data import DataLoader
from .datasets import DeepFakeDataset
from utils.config import GlobalConfig

def get_data_loader(
        ds: DeepFakeDataset, 
        shuffle: bool, is_train: bool=False
    ) -> DataLoader:
    
    return DataLoader(
        ds, batch_size=GlobalConfig.BATCH_SIZE, shuffle=shuffle, 
        num_workers=GlobalConfig.NUM_WORKERS, 
        pin_memory=(GlobalConfig.DEVICE.type == 'cuda' and GlobalConfig.NUM_WORKERS > 0), 
        persistent_workers=(GlobalConfig.NUM_WORKERS > 0 and is_train),
        drop_last=is_train
    )