import torch.nn as nn

def patch_dropout(module: nn.Module, p: float):
    """
    Recursively replace every Dropout or Dropout2d layer in `module` with p=`p`.
    """
    for name, child in module.named_children():
        if isinstance(child, (nn.Dropout, nn.Dropout2d)):
            setattr(module, name, type(child)(p=p, inplace=child.inplace))
        else:
            patch_dropout(child, p)
