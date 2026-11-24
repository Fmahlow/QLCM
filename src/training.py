from __future__ import annotations


class ConvergenceTracker:
    """Shared stopping criterion for every trainable model in this project
    (docs/research_design.md, "Shared methodology"): stop when the moving
    average training loss over the last `window` epochs has improved by
    less than `rel_improvement_threshold` relative to the previous `window`
    epochs, for `patience` consecutive checks in a row (to avoid stopping on
    a single noisy dip), subject to a hard `max_epochs` cap applied
    identically across every model family. Using one rule everywhere is the
    direct fix for v1's problem of picking per-model epoch budgets by
    analogy (e.g. 150 vs. 50 epochs for architecturally different models).
    """

    def __init__(
        self,
        window: int = 10,
        rel_improvement_threshold: float = 0.005,
        patience: int = 3,
        max_epochs: int = 300,
    ):
        self.window = window
        self.rel_improvement_threshold = rel_improvement_threshold
        self.patience = patience
        self.max_epochs = max_epochs
        self.history: list[float] = []
        self._below_threshold_count = 0

    def update(self, epoch_loss: float) -> bool:
        """Record this epoch's loss; return True if training should stop."""
        self.history.append(epoch_loss)
        epoch = len(self.history)
        if epoch >= self.max_epochs:
            return True
        if epoch < 2 * self.window:
            return False

        prev_avg = sum(self.history[-2 * self.window : -self.window]) / self.window
        curr_avg = sum(self.history[-self.window :]) / self.window
        rel_improvement = (prev_avg - curr_avg) / abs(prev_avg) if prev_avg != 0 else 0.0

        if rel_improvement < self.rel_improvement_threshold:
            self._below_threshold_count += 1
        else:
            self._below_threshold_count = 0
        return self._below_threshold_count >= self.patience

    @property
    def epochs_run(self) -> int:
        return len(self.history)

    @property
    def hit_cap(self) -> bool:
        """True if training stopped because of max_epochs, not convergence --
        must be reported explicitly rather than treated as a finished result
        (docs/research_design.md)."""
        return len(self.history) >= self.max_epochs and self._below_threshold_count < self.patience
