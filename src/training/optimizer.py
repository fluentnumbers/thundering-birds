import numpy as np
import torch.optim as optim






def get_optimizer(model, cfg):

    if cfg.training.OPTIMIZER == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.training.BASE_LR,
            weight_decay=cfg.training.WEIGHT_DECAY,
        )
    elif cfg.training.OPTIMIZER == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.training.BASE_LR,
            weight_decay=cfg.training.WEIGHT_DECAY,
        )
    elif cfg.training.OPTIMIZER == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.training.BASE_LR,
            momentum=0.9,
            weight_decay=cfg.training.WEIGHT_DECAY,
        )
    else:
        raise NotImplementedError(f"Optimizer {cfg.training.OPTIMIZER} not implemented")





    return optimizer

def get_optimizer2(model, cfg):
    """Get optimizer with layer-wise learning rates and dynamic scaling.

    The learning rate is scaled based on:
    1. Batch size
    2. Number of classes
    3. Class imbalance (if using weighted sampling)
    """
    # Calculate base learning rate scaling based on batch size
    batch_size_scale = np.sqrt(cfg.training.BATCH_SIZE / 256)

    # Scale based on number of classes (more classes -> slightly lower LR)
    num_classes_scale = np.sqrt(100 / cfg.num_classes)  # normalized to 100 classes

    # Determine if using weighted sampling and adjust accordingly
    using_weighted = cfg.training.SAMPLING_CLASSES_WEIGHTS != "uniform"
    imbalance_scale = 0.7 if using_weighted else 1.0  # Reduce LR with weighted sampling

    # Calculate final base learning rate
    if cfg.training.LR_SCALING:
        base_lr = (
            cfg.training.BASE_LR
            * batch_size_scale
            * num_classes_scale
            * imbalance_scale
        )
    else:
        base_lr = cfg.training.BASE_LR

    # Group parameters by layer depth for layer-wise learning rates
    layer_groups = []

    # Classifier head (final layer) gets lowest LR
    classifier_params = []
    backbone_params = []

    # Separate classifier and backbone parameters
    for name, param in model.named_parameters():
        if "model.model._fc" in name:  # Classifier layer
            classifier_params.append(param)
        else:
            backbone_params.append(param)

    # Add parameter groups with different learning rates
    param_groups = [
        {"params": backbone_params, "lr": base_lr},
        {"params": classifier_params, "lr": base_lr * cfg.training.LR_SCALE_FACTOR},
    ]

    if cfg.training.OPTIMIZER == "Adam":
        optimizer = optim.Adam(
            param_groups,
            weight_decay=cfg.training.WEIGHT_DECAY,
        )
    elif cfg.training.OPTIMIZER == "AdamW":
        optimizer = optim.AdamW(
            param_groups,
            weight_decay=cfg.training.WEIGHT_DECAY,
        )
    elif cfg.training.OPTIMIZER == "SGD":
        optimizer = optim.SGD(
            param_groups,
            momentum=0.9,
            weight_decay=cfg.training.WEIGHT_DECAY,
        )
    else:
        raise NotImplementedError(f"Optimizer {cfg.training.OPTIMIZER} not implemented")

    return optimizer
