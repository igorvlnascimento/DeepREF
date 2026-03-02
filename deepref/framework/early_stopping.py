class EarlyStopping:
    """Counter-based early stopping helper.

    Args:
        patience: Number of epochs without improvement before stopping.
            ``0`` (default) disables early stopping entirely.
    """

    def __init__(self, patience: int = 0) -> None:
        self.patience = patience
        self._counter = 0

    def step(self, improved: bool) -> bool:
        """Advance the counter and return ``True`` when training should stop.

        Args:
            improved: Whether the monitored metric improved this epoch.

        Returns:
            ``True`` if patience has been exhausted; ``False`` otherwise.
        """
        if self.patience == 0:
            return False
        if improved:
            self._counter = 0
        else:
            self._counter += 1
        return self._counter >= self.patience
