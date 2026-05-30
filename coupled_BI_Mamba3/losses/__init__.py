from .task_losses import (
    build_loss, MultiTaskLoss, InfoNCELoss, SupervisedContrastiveLoss,
    RegressionWithDiscreteCE, OrdinalRegressionLoss, BoundaryAwareL1,
    OrdinalCompositeLoss,
)

__all__ = [
    "build_loss", "MultiTaskLoss", "InfoNCELoss", "SupervisedContrastiveLoss",
    "RegressionWithDiscreteCE", "OrdinalRegressionLoss", "BoundaryAwareL1",
    "OrdinalCompositeLoss",
]