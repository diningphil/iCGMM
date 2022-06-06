from pydgn.training.callback.metric import Metric


class iCGMMLoss(Metric):
    @property
    def name(self) -> str:
        return 'iCGMM Loss'

    def __init__(self, use_as_loss: bool=False, reduction: str='mean', use_nodes_batch_size: bool=True):
        super().__init__(use_as_loss, reduction, use_nodes_batch_size)

    # Simply ignore targets
    def forward(self, targets, *outputs):
        likelihood = outputs[2]
        return likelihood

    def on_backward(self, state):
        pass
