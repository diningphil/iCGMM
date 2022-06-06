import os
from pathlib import Path

import torch
from pydgn.static import *
from pydgn.training.engine import TrainingEngine, log
from pydgn.training.event.handler import EventHandler

from util import extend_lists, to_tensor_lists


class GibbsSamplingEngine(TrainingEngine):

    def _train(self, loader):
        """
        Gibbs sampling with parameters' update. The only difference with the superclass
        method is that set_eval_mode() is not called here
        """
        self._dispatch(EventHandler.ON_TRAINING_EPOCH_START, self.state)

        self._loop(loader)

        self._dispatch(EventHandler.ON_TRAINING_EPOCH_END, self.state)

        assert self.state.epoch_loss is not None
        if self.score_fun is not None:
            loss, score, data_list = self.state.epoch_loss, self.state.epoch_score, self.state.epoch_data_list
        else:
            loss = self.state.epoch_loss
            score = loss
            data_list = self.state.epoch_data_list
        return loss, score, data_list

    def infer(self, loader, set):
        """
        Gibbs sampling without parameters' update. The only difference with the superclass
        method is that set_eval_mode() is not called here
        """
        self.state.update(set=set)

        self._dispatch(EventHandler.ON_EVAL_EPOCH_START, self.state)

        with torch.no_grad():
            self._loop(loader)  # state has been updated

        self._dispatch(EventHandler.ON_EVAL_EPOCH_END, self.state)

        assert self.state.epoch_loss is not None
        loss, score, data_list = self.state.epoch_loss, self.state.epoch_score, self.state.epoch_data_list

        # Add the main loss we want to return as a special key
        main_loss_name = self.loss_fun.get_main_metric_name()
        loss[MAIN_LOSS] = loss[main_loss_name]

        # Add the main score we want to return as a special key
        # Needed by the experimental evaluation framework
        if self.score_fun is not None:
            main_score_name = self.score_fun.get_main_metric_name()
            score[MAIN_SCORE] = score[main_score_name]
            return loss, score, data_list
        else:
            loss[MAIN_SCORE] = loss[main_loss_name]
            return loss, loss, data_list

    def train(self, train_loader, validation_loader=None, test_loader=None, max_epochs=100, zero_epoch=False,
              logger=None):
        """
        Trains the model by Gibbs sampling
        :param train_loader: the DataLoader associated with training data
        :param validation_loader: the DataLoader associated with validation data, if any
        :param test_loader:  the DataLoader associated with test data, if any
        :param max_epochs: maximum number of training Gibbs sampling iterations
        :param zero_epoch: if True, starts again from epoch 0 and resets optimizer and scheduler states.
        :param logger: a log.Logger for logging purposes
        :return: a tuple (train_loss, train_score, train_embeddings, validation_loss, validation_score, validation_embeddings, test_loss, test_score, test_embeddings)
        """
        if self.early_stopper is not None:
            print(f'WARNING: EARLY STOPPING NOT SUPPORTED')
            log(f'WARNING: EARLY STOPPING NOT SUPPORTED', logger)

        try:
            # Initialize variables
            val_loss, val_score, val_embeddings_tuple = None, None, None
            test_loss, test_score, test_embeddings_tuple = None, None, None

            self.set_device()

            # Restore training from last checkpoint if possible!
            ckpt_filename = Path(self.exp_path, LAST_CHECKPOINT_FILENAME)
            best_ckpt_filename = Path(self.exp_path, BEST_CHECKPOINT_FILENAME)
            if os.path.exists(ckpt_filename):
                self._restore_last(ckpt_filename, best_ckpt_filename, zero_epoch)
                log(f'START AGAIN FROM EPOCH {self.state.initial_epoch}', logger)

            self._dispatch(EventHandler.ON_FIT_START, self.state)

            # Perform a separate gibbs sampling loop for each loader
            # Loop over the entire dataset dataset

            self.state.update(set=TRAINING)
            # In Gibbs Sampling, setting train/eval here serves to
            # initialize accumulators
            self.set_training_mode()
            epoch = self.state.initial_epoch
            for epoch in range(self.state.initial_epoch, max_epochs):
                self.state.update(epoch=epoch)
                self.state.update(return_node_embeddings=False)

                self._dispatch(EventHandler.ON_EPOCH_START, self.state)

                # each loss/score is one epoch "behind" because on_backward
                # has not been called already when scores/losses where
                # computed
                # embeddings will be available only after the last gibbs sampling iteration
                train_loss, train_score, train_embeddings_tuple = self._train(train_loader)

                # Update state with epoch results (plotter will plot what's inside here)
                epoch_results = {
                    LOSSES: {},
                    SCORES: {}
                }

                epoch_results[LOSSES].update({f'{TRAINING}_{k}': v for k, v in train_loss.items()})
                epoch_results[SCORES].update({f'{TRAINING}_{k}': v for k, v in train_score.items()})

                # Update state with the result of this epoch
                self.state.update(epoch_results=epoch_results)

                # We can apply early stopping here
                self._dispatch(EventHandler.ON_EPOCH_END, self.state)

                # Log performances
                if epoch % self.evaluate_every == 0 or epoch == 1:
                    msg = f'Epoch: {epoch}, TR loss: {train_loss} TR score: {train_score}'
                    log(msg, logger)

            # Needed to indicate that training has ended
            self.state.update(stop_training=True)

            self._dispatch(EventHandler.ON_FIT_END, self.state)

            # INFERENCE WITH A SINGLE ITERATION
            max_epochs = 1

            self.state.update(set=TRAINING)
            # In Gibbs Sampling, setting train/eval here serves to
            # initialize accumulators
            self.set_eval_mode()
            for epoch in range(max_epochs):
                self.state.update(epoch=epoch)
                self.state.update(return_node_embeddings=epoch == max_epochs - 1)

                self._dispatch(EventHandler.ON_EPOCH_START, self.state)

                # embeddings will be available only after the last gibbs sampling iteration
                train_loss, train_score, train_embeddings_tuple = self.infer(train_loader, TRAINING)

                # Update state with epoch results (plotter will plot what's inside here)
                epoch_results = {
                    LOSSES: {},
                    SCORES: {}
                }

                epoch_results[LOSSES].update({f'{TRAINING}_{k}': v for k, v in train_loss.items()})
                epoch_results[SCORES].update({f'{TRAINING}_{k}': v for k, v in train_score.items()})

                # Update state with the result of this epoch
                self.state.update(epoch_results=epoch_results)

                # We can apply early stopping here
                self._dispatch(EventHandler.ON_EPOCH_END, self.state)

                # Log performances
                if epoch % self.evaluate_every == 0 or epoch == 1:
                    msg = f'Epoch: {epoch}, TR loss: {train_loss} TR score: {train_score}'
                    log(msg, logger)

            if validation_loader is not None:
                self.state.update(set=VALIDATION)
                # In Gibbs Sampling, setting train/eval here serves to
                # initialize accumulators
                self.set_eval_mode()
                for epoch in range(max_epochs):
                    self.state.update(epoch=epoch)

                    self._dispatch(EventHandler.ON_EPOCH_START, self.state)

                    # embeddings will be available only after the last gibbs sampling iteration
                    val_loss, val_score, val_embeddings_tuple = self.infer(validation_loader, VALIDATION)

                    # Update state with epoch results (plotter will plot what's inside here)
                    epoch_results = {
                        LOSSES: {},
                        SCORES: {}
                    }

                    epoch_results[LOSSES].update({f'{VALIDATION}_{k}': v for k, v in val_loss.items()})
                    epoch_results[SCORES].update({f'{VALIDATION}_{k}': v for k, v in val_score.items()})

                    # Update state with the result of this epoch
                    self.state.update(epoch_results=epoch_results)

                    # We can apply early stopping here
                    self._dispatch(EventHandler.ON_EPOCH_END, self.state)

                    # Log performances
                    if epoch % self.evaluate_every == 0 or epoch == 1:
                        msg = f'Epoch: {epoch}, VL loss: {val_loss} VL score: {val_score}'
                        log(msg, logger)

            if test_loader is not None:
                self.state.update(set=TEST)
                # In Gibbs Sampling, setting train/eval here serves to
                # initialize accumulators
                self.set_eval_mode()
                for epoch in range(max_epochs):
                    self.state.update(epoch=epoch)

                    self._dispatch(EventHandler.ON_EPOCH_START, self.state)

                    # embeddings will be available only after the last gibbs sampling iteration
                    test_loss, test_score, test_embeddings_tuple = self.infer(test_loader, TEST)

                    # Update state with epoch results (plotter will plot what's inside here)
                    epoch_results = {
                        LOSSES: {},
                        SCORES: {}
                    }

                    epoch_results[LOSSES].update({f'{TEST}_{k}': v for k, v in test_loss.items()})
                    epoch_results[SCORES].update({f'{TEST}_{k}': v for k, v in test_score.items()})

                    # Update state with the result of this epoch
                    self.state.update(epoch_results=epoch_results)

                    # We can apply early stopping here
                    self._dispatch(EventHandler.ON_EPOCH_END, self.state)

                    # Log performances
                    if epoch % self.evaluate_every == 0 or epoch == 1:
                        msg = f'Epoch: {epoch}, TE loss: {test_loss} TE score: {test_score}'
                        log(msg, logger)

            self.state.update(return_node_embeddings=False)

            log(f'Chosen is TR loss: {train_loss} TR score: {train_score}, VL loss: {val_loss} VL score: {val_score} '
                f'TE loss: {test_loss} TE score: {test_score}', logger)

            self.state.update(set=None)

        except Exception as e:
            log(f'CATCHED EXCEPTION, exiting gracefully...', logger)
            report = self.profiler.report()
            log(str(e), logger)
            log(report, logger)
            raise e

        # Log profile results
        report = self.profiler.report()
        log(report, logger)

        return train_loss, train_score, train_embeddings_tuple, \
               val_loss, val_score, val_embeddings_tuple, \
               test_loss, test_score, test_embeddings_tuple


class IncrementalGibbsSamplingEngine(GibbsSamplingEngine):

    def __init__(self, engine_callback, model, loss, optimizer, scorer=None,
                 scheduler=None, early_stopper=None, gradient_clipper=None, device='cpu', plotter=None, exp_path=None,
                 evaluate_every=1, store_last_checkpoint=False):
        super().__init__(engine_callback, model, loss, optimizer, scorer,
                         scheduler, early_stopper, gradient_clipper, device,
                         plotter, exp_path, evaluate_every, store_last_checkpoint)

    def _to_list(self, data_list, embeddings, batch, edge_index, y, linkpred=False):

        if isinstance(embeddings, tuple):
            embeddings = tuple([e.detach().cpu() if e is not None else None for e in embeddings])
        elif isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu()
        else:
            raise NotImplementedError('Embeddings not understood, should be Tensor or Tuple of Tensors')

        data_list = extend_lists(data_list, to_tensor_lists(embeddings, batch, edge_index))
        return data_list
