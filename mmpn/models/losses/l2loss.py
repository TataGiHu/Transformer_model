import torch
import torch.nn as nn

from ..builder import LOSSES
from .utils import weighted_loss



@weighted_loss
def l2_loss(pred, target):
    """L2 loss

    Args:
        pred ([type]): [description]
        target ([type]): [description]

    Returns:
        [type]: [description]
    """    
    assert pred.size() == target.size() and target.numel() > 0
    loss = torch.square(pred - target)
    return loss
  
    
    



@LOSSES.register_module()
class L2Loss(nn.Module):
    """L2 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(L2Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * l2_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox



