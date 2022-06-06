import os

import torch
from pydgn.experiment.util import s2c
from pydgn.static import LOSS, SCORE
from torch_geometric.data import Data
from torch_geometric.data import DataLoader

from icgmm_incremental_task import iCGMMTask


# This works with graph classification only
class ClassifieriCGMMTask(iCGMMTask):

    def run_valid(self, dataset_getter, logger):
        """
        This function returns the training and validation or test accuracy
        :return: (training accuracy, validation/test accuracy)
        """

        # Necessary info to give a unique name to the dataset (some hyper-params like epochs are assumed to be fixed)
        embeddings_folder = self.model_config.layer_config['embeddings_folder']
        gamma = self.model_config.layer_config['gamma'] if 'gamma' in self.model_config.layer_config else \
        self.model_config.layer_config['gamma_prior_params']
        alpha = self.model_config.layer_config['alpha'] if 'alpha' in self.model_config.layer_config else \
        self.model_config.layer_config['alpha_prior_params']
        epochs = self.model_config.layer_config['epochs']
        sample_neighboring_macrostate = self.model_config.layer_config['sample_neighboring_macrostate']

        max_layers = self.model_config.layer_config['max_layers']
        layers = self.model_config.layer_config['layers']
        unibigram = self.model_config.layer_config['unibigram']
        aggregation = self.model_config.layer_config['aggregation']
        outer_k = dataset_getter.outer_k
        inner_k = dataset_getter.inner_k
        # ====

        hyperprior = self.model_config.layer_config['emission_distribution']['prior_params']
        hyperprior_str = '_'.join([f'{k}_{v}' for k, v in hyperprior.items()])

        base_path = os.path.join(embeddings_folder, dataset_getter.dataset_name,
                                 f'{max_layers}_{epochs}_{unibigram}_{aggregation}_{hyperprior_str}_{alpha}_{gamma}_{sample_neighboring_macrostate}_{outer_k + 1}_{inner_k + 1}')

        all_C = torch.load(base_path + '_allC.torch')
        features_to_use = sum([C + C * C for C in all_C[:layers]])
        # assumes unibigram is true
        assert unibigram == True

        train_out_emb = torch.load(base_path + '_train.torch')[:, :features_to_use]
        val_out_emb = torch.load(base_path + '_val.torch')[:, :features_to_use]
        train_out_emb = torch.reshape(train_out_emb, (train_out_emb.shape[0], -1))
        val_out_emb = torch.reshape(val_out_emb, (val_out_emb.shape[0], -1))

        # Recover the targets
        fake_train_loader = dataset_getter.get_inner_train(batch_size=1, shuffle=False)
        fake_val_loader = dataset_getter.get_inner_val(batch_size=1, shuffle=False)
        train_y = [el.y for el in fake_train_loader.dataset]
        val_y = [el.y for el in fake_val_loader.dataset]
        arbitrary_logic_batch_size = self.model_config.layer_config['arbitrary_function_config']['batch_size']
        arbitrary_logic_shuffle = self.model_config.layer_config['arbitrary_function_config']['shuffle'] \
            if 'shuffle' in self.model_config.layer_config['arbitrary_function_config'] else True

        # build data lists
        train_list = [Data(x=train_out_emb[i].unsqueeze(0), y=train_y[i]) for i in range(train_out_emb.shape[0])]
        val_list = [Data(x=val_out_emb[i].unsqueeze(0), y=val_y[i]) for i in range(val_out_emb.shape[0])]
        train_loader = DataLoader(train_list, batch_size=arbitrary_logic_batch_size, shuffle=arbitrary_logic_shuffle)
        val_loader = DataLoader(val_list, batch_size=arbitrary_logic_batch_size, shuffle=arbitrary_logic_shuffle)

        # Instantiate the Dataset
        dim_features = train_out_emb.shape[1]
        dim_target = dataset_getter.get_dim_target()

        config = self.model_config.layer_config['arbitrary_function_config']
        device = config['device']

        readout_class = s2c(config['predictor'])
        model = readout_class(dim_node_features=dim_features,
                                dim_edge_features=0,
                                dim_target=dim_target,
                                config=config)

        predictor_wrapper = self._create_engine(config, model, device, evaluate_every=self.model_config.evaluate_every)

        train_loss, train_score, _, \
        val_loss, val_score, _, \
        _, _, _ = predictor_wrapper.train(train_loader=train_loader,
                                          validation_loader=val_loader,
                                          test_loader=None,
                                          max_epochs=config['epochs'],
                                          logger=logger)

        train_res = {LOSS: train_loss, SCORE: train_score}
        val_res = {LOSS: val_loss, SCORE: val_score}
        return train_res, val_res

    def run_test(self, dataset_getter, logger):
        """
        This function returns the training and test accuracy. DO NOT USE THE TEST FOR ANY REASON
        :return: (training accuracy, test accuracy)
        """

        # Necessary info to give a unique name to the dataset (some hyper-params like epochs are assumed to be fixed)
        embeddings_folder = self.model_config.layer_config['embeddings_folder']
        gamma = self.model_config.layer_config['gamma'] if 'gamma' in self.model_config.layer_config else \
        self.model_config.layer_config['gamma_prior_params']
        alpha = self.model_config.layer_config['alpha'] if 'alpha' in self.model_config.layer_config else \
        self.model_config.layer_config['alpha_prior_params']
        epochs = self.model_config.layer_config['epochs']
        sample_neighboring_macrostate = self.model_config.layer_config['sample_neighboring_macrostate']

        max_layers = self.model_config.layer_config['max_layers']
        layers = self.model_config.layer_config['layers']
        unibigram = self.model_config.layer_config['unibigram']
        aggregation = self.model_config.layer_config['aggregation']
        outer_k = dataset_getter.outer_k
        inner_k = dataset_getter.inner_k
        if inner_k is None:  # workaround the "safety" procedure of evaluation protocol, but we will not do anything wrong.
            dataset_getter.set_inner_k(0)
            inner_k = 0  # pick the split of the first inner fold
        # ====

        # NOTE: We reload the associated inner train and val splits, using the outer_test for assessment.
        # This is slightly different from standard exps, where we compute a different outer train-val split, but it should not change things much.

        hyperprior = self.model_config.layer_config['emission_distribution']['prior_params']
        hyperprior_str = '_'.join([f'{k}_{v}' for k, v in hyperprior.items()])
        base_path = os.path.join(embeddings_folder, dataset_getter.dataset_name,
                                 f'{max_layers}_{epochs}_{unibigram}_{aggregation}_{hyperprior_str}_{alpha}_{gamma}_{sample_neighboring_macrostate}_{outer_k + 1}_{inner_k + 1}')

        all_C = torch.load(base_path + '_allC.torch')
        features_to_use = sum([C + C * C for C in all_C[:layers]])
        # assumes unibigram is true
        assert unibigram == True

        train_out_emb = torch.load(base_path + '_train.torch')[:, :features_to_use]
        val_out_emb = torch.load(base_path + '_val.torch')[:, :features_to_use]
        test_out_emb = torch.load(base_path + '_test.torch')[:, :features_to_use]
        train_out_emb = torch.reshape(train_out_emb, (train_out_emb.shape[0], -1))
        val_out_emb = torch.reshape(val_out_emb, (val_out_emb.shape[0], -1))
        test_out_emb = torch.reshape(test_out_emb, (test_out_emb.shape[0], -1))

        # Recover the targets
        fake_train_loader = dataset_getter.get_inner_train(batch_size=1, shuffle=False)
        fake_val_loader = dataset_getter.get_inner_val(batch_size=1, shuffle=False)
        fake_test_loader = dataset_getter.get_outer_test(batch_size=1, shuffle=False)
        train_y = [el.y for el in fake_train_loader.dataset]
        val_y = [el.y for el in fake_val_loader.dataset]
        test_y = [el.y for el in fake_test_loader.dataset]
        arbitrary_logic_batch_size = self.model_config.layer_config['arbitrary_function_config']['batch_size']
        arbitrary_logic_shuffle = self.model_config.layer_config['arbitrary_function_config']['shuffle'] \
            if 'shuffle' in self.model_config.layer_config['arbitrary_function_config'] else True

        # build data lists
        train_list = [Data(x=train_out_emb[i].unsqueeze(0), y=train_y[i]) for i in range(train_out_emb.shape[0])]
        val_list = [Data(x=val_out_emb[i].unsqueeze(0), y=val_y[i]) for i in range(val_out_emb.shape[0])]
        test_list = [Data(x=test_out_emb[i].unsqueeze(0), y=test_y[i]) for i in range(test_out_emb.shape[0])]
        train_loader = DataLoader(train_list, batch_size=arbitrary_logic_batch_size, shuffle=arbitrary_logic_shuffle)
        val_loader = DataLoader(val_list, batch_size=arbitrary_logic_batch_size, shuffle=arbitrary_logic_shuffle)
        test_loader = DataLoader(test_list, batch_size=arbitrary_logic_batch_size, shuffle=arbitrary_logic_shuffle)

        # Instantiate the Dataset
        dim_features = train_out_emb.shape[1]
        dim_target = dataset_getter.get_dim_target()

        config = self.model_config.layer_config['arbitrary_function_config']
        device = config['device']

        readout_class = s2c(config['predictor'])
        model = readout_class(dim_node_features=dim_features,
                                dim_edge_features=0,
                                dim_target=dim_target,
                                config=config)

        predictor_wrapper = self._create_engine(config, model, device, evaluate_every=self.model_config.evaluate_every)

        train_loss, train_score, _, \
        val_loss, val_score, _, \
        test_loss, test_score, _ = predictor_wrapper.train(train_loader=train_loader,
                                                           validation_loader=val_loader,
                                                           test_loader=test_loader,
                                                           max_epochs=config['epochs'],
                                                           logger=logger)

        train_res = {LOSS: train_loss, SCORE: train_score}
        val_res = {LOSS: val_loss, SCORE: val_score}
        test_res = {LOSS: test_loss, SCORE: test_score}
        return train_res, val_res, test_res
