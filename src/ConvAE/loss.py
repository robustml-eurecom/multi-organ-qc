import torch
import torch.nn as nn

class Loss():
    def __init__(self, loss_functions, settling_epochs_BKGDLoss, settling_epochs_BKMSELoss):
        """
        A custom loss class for calculating multiple loss functions with optional settling epochs.

        Args:
            loss_functions (dict): A dictionary of loss functions, where keys are loss function names and values are PyTorch loss modules.
            settling_epochs_BKGDLoss (int): Number of epochs to settle before applying 'BKGDLoss'.
            settling_epochs_BKMSELoss (int): Number of epochs to settle before applying 'BKMSELoss'.

        Example:
            >>> loss_functions = {
            ...     'MSELoss': nn.MSELoss(),
            ...     'BKMSELoss': BKMSELoss(),
            ...     'BKGDLoss': BKGDLoss(),
            ...     'GDLoss': GDLoss()
            ... }
            >>> settling_epochs_BKGDLoss = 10
            >>> settling_epochs_BKMSELoss = 5
            >>> loss = Loss(loss_functions, settling_epochs_BKGDLoss, settling_epochs_BKMSELoss)
            >>> prediction = torch.rand(16, 3, 128, 128)
            >>> target = torch.rand(16, 3, 128, 128)
            >>> epoch = 3
            >>> result = loss(prediction, target, epoch, validation=False)
        """
        
        self.loss_functions = loss_functions
        self.settling_epochs_BKGDLoss = settling_epochs_BKGDLoss
        self.settling_epochs_BKMSELoss = settling_epochs_BKMSELoss

    def __call__(self, prediction, target, epoch, validation=False):
        contributes = {}

        for loss_name, loss_fn in self.loss_functions.items():
            contributes[loss_name] = loss_fn(prediction, target)

        if "BKGDLoss" in contributes and epoch < self.settling_epochs_BKGDLoss:
            bgd_loss = self.loss_functions["BKGDLoss"]
            contributes["BKGDLoss"] += bgd_loss(prediction[:, 1:], target[:, 1:])
        if "BKMSELoss" in contributes and epoch < self.settling_epochs_BKMSELoss:
            bkmse_loss = self.loss_functions["BKMSELoss"]
            contributes["BKMSELoss"] += bkmse_loss(prediction[:, 1:], target[:, 1:])

        contributes["Total"] = sum(contributes.values())

        if validation:
            return {k: v.item() for k, v in contributes.items()}
        else:
            return contributes["Total"]

class BKMSELoss(nn.Module):
    def __init__(self):
        super(BKMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, prediction, target):
        return self.mse_loss(prediction, target)

class BKGDLoss(nn.Module):
    def __init__(self):
        super(BKGDLoss, self).__init__()

    def forward(self, prediction, target):
        intersection = torch.sum(prediction * target, dim=(1, 2, 3))
        cardinality = torch.sum(prediction + target, dim=(1, 2, 3))
        dice_score = 2. * intersection / (cardinality + 1e-6)
        return torch.mean(1 - dice_score)

class GDLoss(nn.Module):
    def __init__(self):
        super(GDLoss, self).__init__()

    def forward(self, prediction, target):
        intersection = torch.sum(prediction * target, dim=(0, 2, 3))
        cardinality = torch.sum(prediction + target, dim=(0, 2, 3))
        dice_score = 2. * intersection / (cardinality + 1e-6)
        return torch.mean(1 - dice_score)

