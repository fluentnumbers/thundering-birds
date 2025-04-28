import torch.nn as nn

from src.training.losses import AsymmetricLossMultiLabel, HierarchicalBCELoss, DynamicWeightedBCELoss


def get_criterion(cfg):

    if cfg.training.CRITERION == "BCEWithLogitsLoss":
        if hasattr(cfg.training, 'LOSS_WEIGHTING') and cfg.training.LOSS_WEIGHTING == "dynamic":
            criterion = DynamicWeightedBCELoss(
                num_classes=cfg.num_classes,
                momentum=cfg.training.LOSS_MOMENTUM,
                temperature=cfg.training.LOSS_TEMPERATURE,
                min_weight=cfg.training.LOSS_MIN_WEIGHT
            )
        else:
            criterion = nn.BCEWithLogitsLoss()
    elif cfg.training.CRITERION == "AsymmetricLossMultiLabel":
        criterion = AsymmetricLossMultiLabel(
            gamma_neg=4,
            gamma_pos=1,
            clip=0.05,
            eps=1e-8,
            disable_torch_grad_focal_loss=False,
            reduction="mean",
        )
    elif cfg.training.CRITERION == "HierarchicalBCELoss":
        criterion = HierarchicalBCELoss(
            primary_weight=cfg.training.PRIMARY_WEIGHT,
            secondary_weight=cfg.training.SECONDARY_WEIGHT,
        )
    elif cfg.training.CRITERION == "CELoss":
        criterion = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(f"Criterion {cfg.training.CRITERION} not implemented")

    return criterion
