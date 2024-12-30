import torch


class TweedieLoss(torch.nn.Module):
    """
    Tweedie loss.

    Tweedie regression with log-link. It might be useful, e.g., for modeling total
    loss in insurance, or for any target that might be tweedie-distributed.
    """

    def __init__(self, p: float = 1.5):
        """
        Args:
            p (float, optional): tweedie variance power which is greater equal
                1.0 and smaller 2.0. Close to ``2`` shifts to
                Gamma distribution and close to ``1`` shifts to Poisson distribution.
                Defaults to 1.5.
            reduction (str, optional): How to reduce the loss. Defaults to "mean".
        """
        super().__init__()
        assert 1 <= p < 2, "p must be in range [1, 2]"
        self.p = p

    def forward(self, y_pred, y_true):
        # clip y_pred to be positive
        y_pred = torch.clip(y_pred, min=0.0001)
        a = y_true * torch.pow(y_pred, 1 - self.p) / (1 - self.p)
        b = torch.pow(y_pred, 2 - self.p) / (2 - self.p)
        loss = -a + b
        return loss


if __name__ == "__main__":
    import torch

    def test_tweedie_loss():
        y_true = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0], dtype=torch.float32)
        y_pred = torch.tensor([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0], dtype=torch.float32)
        loss_fn = TweedieLoss(p=1.5)
        loss = loss_fn(y_pred, y_true)
        print("Tweedie Loss:", loss)

    test_tweedie_loss()
