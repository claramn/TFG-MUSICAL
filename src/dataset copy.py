from torch.utils.data import DataLoader, Subset
from src.dataset import NSynth

STRING_ID = 8  # string

def only_string_subset(partition):
    ds = NSynth(partition=partition)
    idx = [
        i for i, k in enumerate(ds._keys)
        if int(ds._metadata[k]["one_hot_instrument"].argmax()) == STRING_ID
    ]
    return Subset(ds, idx)

train_ds = only_string_subset("training")
valid_ds = only_string_subset("validation")

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
