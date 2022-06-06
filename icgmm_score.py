import torch
from pydgn.training.callback.metric import Metric


class iCGMMCompleteLikelihoodScore(Metric):
    @property
    def name(self) -> str:
        return 'iCGMM Complete Log Likelihood'

    def forward(self, targets, *outputs, batch_loss_extra):
        return outputs[2]


class iCGMMCompleteLikelihoodScore2(iCGMMCompleteLikelihoodScore):
    @property
    def name(self) -> str:
        return 'iCGMM Complete Log Likelihood 2'

    def forward(self, targets, *outputs, batch_loss_extra):
        return outputs[3]


class iCGMMCompleteLikelihoodScore3(iCGMMCompleteLikelihoodScore):
    @property
    def name(self) -> str:
        return 'iCGMM Complete Log Likelihood 3'

    def forward(self, targets, *outputs, batch_loss_extra):
        return outputs[4]


class iCGMMCurrentStates(Metric):
    @property
    def name(self) -> str:
        return 'iCGMM Current States'

    def on_training_batch_end(self, state):
        pass

    def on_training_epoch_end(self, state):
        state.update(epoch_score={self.name: state.model.Ccurr.clone().detach()})

    def on_eval_epoch_end(self, state):
        state.update(epoch_score={self.name: state.model.Ccurr.clone().detach()})

    def on_eval_batch_end(self, state):
        pass

    def forward(self, targets, *outputs, batch_loss_extra):
        return torch.tensor(0.)


class iCGMMCurrentAlpha(Metric):
    @property
    def name(self) -> str:
        return 'iCGMM Current Alpha'

    def on_training_epoch_end(self, state):
        state.update(epoch_score={self.name: state.model.alpha.clone().detach()})

    def on_eval_epoch_end(self, state):
        state.update(epoch_score={self.name: state.model.alpha.clone().detach()})

    def forward(self, targets, *outputs, batch_loss_extra):
        return torch.tensor(0.)


class iCGMMCurrentGamma(Metric):
    @property
    def name(self) -> str:
        return 'iCGMM Current Gamma'

    def on_training_epoch_end(self, state):
        state.update(epoch_score={self.name: state.model.gamma.clone().detach()})

    def on_eval_epoch_end(self, state):
        state.update(epoch_score={self.name: state.model.gamma.clone().detach()})

    def forward(self, targets, *outputs, batch_loss_extra):
        return torch.tensor(0.)
