import numpy as np
from scipy.spatial.distance import directed_hausdorff

class Metrics():
    def __init__(self, keys):
        """
        A custom metrics class for calculating Dice Coefficient (DC) and Hausdorff Distance (HD) metrics for multiple classes.

        Args:
            keys (list): List of class labels or keys for which metrics will be computed.

        Example:
            >>> keys = ['BK, 'K1', 'K2']
            >>> metrics = Metrics(keys)
            >>> prediction = np.random.rand(3, 128, 128)
            >>> target = np.random.rand(3, 128, 128)
            >>> results = metrics(prediction, target)
        """
        self.keys = keys

    def dice_coefficient(self, prediction, target):
        intersection = np.sum(prediction * target)
        union = np.sum(prediction) + np.sum(target)
        return (2.0 * intersection) / (union + 1e-6)

    def hausdorff_distance(self, prediction, target):
        try:
            return directed_hausdorff(target, prediction)[0]
        except Exception:
            return np.nan

    def __call__(self, prediction, target):
        metrics = {}
        for c, key in enumerate(self.keys):
            ref = np.copy(target)
            pred = np.copy(prediction)

            ref = np.where(ref != c, 0, 1)
            pred = np.where(pred != c, 0, 1)

            dc = self.dice_coefficient(pred, ref)
            hd = self.hausdorff_distance(pred, ref)

            metrics[f'{key}_dc'] = dc
            metrics[f'{key}_hd'] = hd

        return metrics
