import os
import shutil

import torch
from pydgn.static import LOSS, SCORE

from icgmm_incremental_task import iCGMMTask


# This works with graph classification only
class EmbeddingiCGMMTask(iCGMMTask):

    def run_valid(self, dataset_getter, logger):
        """
        This function returns the training and validation or test accuracy
        :return: (training accuracy, validation/test accuracy)
        """

        batch_size = self.model_config.layer_config['batch_size']
        shuffle = self.model_config.layer_config['shuffle'] \
            if 'shuffle' in self.model_config.layer_config else True

        layers = []
        l_prec = self.model_config.layer_config['previous_layers_to_use'].split(',')
        concatenate_axis = self.model_config.layer_config['concatenate_on_axis']
        max_layers = self.model_config.layer_config['max_layers']
        assert concatenate_axis != 0, 'You cannot concat on the first axis for design reasons.'

        dict_per_layer = []

        stop = False
        depth = 1
        last_C = 1  # to consider first layer as HDP with one layer -1
        all_C = []
        while not stop and depth <= max_layers:

            # Change exp path to allow Stop & Resume
            self.exp_path = os.path.join(self.root_exp_path, f'layer_{depth}')
            # Test output is the last one to be computed
            test_folder = os.path.join(self.root_exp_path, 'outputs', 'test')
            if os.path.exists(os.path.join(test_folder, f'graph_output_{depth}.pt')) and \
                    os.path.exists(os.path.join(test_folder, f'vertex_other_output_{depth}.pt')):
                # print("skip layer", depth)
                # Find last C
                prev_outputs_to_consider = [depth]
                test_out = self._create_extra_dataset(prev_outputs_to_consider, mode='test', depth=depth)
                vo_outs = test_out[0].vo_outs
                C = vo_outs.shape[-1]
                # print(f'C at layer {depth} was {C}')

                last_C = C
                all_C.append(last_C)

                depth += 1
                continue

            # load output will concatenate in reverse order
            prev_outputs_to_consider = [(depth - int(x)) for x in l_prec if (depth - int(x)) > 0]

            train_out = self._create_extra_dataset(prev_outputs_to_consider, mode='train', depth=depth)
            val_out = self._create_extra_dataset(prev_outputs_to_consider, mode='validation', depth=depth)
            train_loader = dataset_getter.get_inner_train(batch_size=batch_size, shuffle=False, extra=train_out)
            val_loader = dataset_getter.get_inner_val(batch_size=batch_size, shuffle=False, extra=val_out)

            # ==== # WARNING: WE ARE JUSTPRECOMPUTING OUTER_TEST EMBEDDINGS FOR SUBSEQUENT CLASSIFIERS
            # WE ARE NOT TRAINING ON TEST (EVEN THOUGH UNSUPERVISED)
            # ==== #

            test_out = self._create_extra_dataset(prev_outputs_to_consider, mode='test', depth=depth)
            test_loader = dataset_getter.get_outer_test(batch_size=batch_size, shuffle=False, extra=test_out)

            # ==== #

            # Instantiate the Dataset
            dim_node_features = dataset_getter.get_dim_node_features()
            dim_edge_features = dataset_getter.get_dim_edge_features()
            dim_target = dataset_getter.get_dim_target()

            # set up J as the chosen number of states C in previous layer
            self.model_config.layer_config['J'] = last_C
            # Instantiate the Model
            new_layer = self.create_incremental_model(dim_node_features, dim_edge_features, dim_target, depth,
                                                      prev_outputs_to_consider)

            # Instantiate the wrapper (it handles the training loop and the inference phase by abstracting the specifics)
            incremental_training_wrapper = self.create_incremental_engine(new_layer)

            train_loss, train_score, train_out, \
            val_loss, val_score, val_out, \
            _, _, test_out = incremental_training_wrapper.train(train_loader=train_loader,
                                                                validation_loader=val_loader,
                                                                test_loader=test_loader,
                                                                max_epochs=self.model_config.layer_config['epochs'],
                                                                logger=logger)

            last_C = new_layer.Ccurr.detach().item()
            all_C.append(last_C)

            for loader, out, mode in [(train_loader, train_out, 'train'), (val_loader, val_out, 'validation'),
                                      (test_loader, test_out, 'test')]:
                v_out, e_out, g_out, vo_out, eo_out, go_out = out

                # Reorder outputs, which are produced in shuffled order, to the original arrangement of the dataset.
                v_out, e_out, g_out, vo_out, eo_out, go_out = self._reorder_shuffled_objects(v_out, e_out, g_out,
                                                                                             vo_out, eo_out, go_out,
                                                                                             loader)

                # Store outputs
                self._store_outputs(mode, depth, v_out, e_out, g_out, vo_out, eo_out, go_out)

            depth += 1

        # NOW LOAD ALL EMBEDDINGS AND STORE THE EMBEDDINGS DATASET ON a torch file.

        # Consider all previous layers now, i.e. gather all the embeddings
        prev_outputs_to_consider = [l for l in range(1, depth + 1)]
        prev_outputs_to_consider.reverse()  # load output will concatenate in reverse order

        # Retrieve only the graph embeddings to save memory.
        # In CGMM classfication task (see other experiment file), I will ignore the outer val and reuse the inner val as validation, as I cannot use the splitter.
        train_out = self._create_extra_dataset(prev_outputs_to_consider, mode='train', depth=depth, only_g=True)
        val_out = self._create_extra_dataset(prev_outputs_to_consider, mode='validation', depth=depth, only_g=True)
        test_out = self._create_extra_dataset(prev_outputs_to_consider, mode='test', depth=depth, only_g=True)

        # Necessary info to give a unique name to the dataset (some hyper-params like epochs are assumed to be fixed)
        embeddings_folder = self.model_config.layer_config['embeddings_folder']
        hyperprior = self.model_config.layer_config['emission_distribution']['prior_params']
        epochs = self.model_config.layer_config['epochs']
        gamma = self.model_config.layer_config['gamma'] if 'gamma' in self.model_config.layer_config else \
        self.model_config.layer_config['gamma_prior_params']
        alpha = self.model_config.layer_config['alpha'] if 'alpha' in self.model_config.layer_config else \
        self.model_config.layer_config['alpha_prior_params']
        sample_neighboring_macrostate = self.model_config.layer_config['sample_neighboring_macrostate']

        max_layers = self.model_config.layer_config['max_layers']
        unibigram = self.model_config.layer_config['unibigram']
        aggregation = self.model_config.layer_config['aggregation']
        outer_k = dataset_getter.outer_k
        inner_k = dataset_getter.inner_k
        # ====

        if not os.path.exists(os.path.join(embeddings_folder, dataset_getter.dataset_name)):
            os.makedirs(os.path.join(embeddings_folder, dataset_getter.dataset_name))

        unigram_dim = sum(all_C)
        assert unibigram == True
        # with iCGMM (variable C) it's not easy to extract unigrams, here you already have conatenated all representations across layers
        # I cannot simply reshape tensors
        # concatenate_on_axis has been set to -1 in the config file.
        for unib in [True]:
            hyperprior = self.model_config.layer_config['emission_distribution']['prior_params']
            hyperprior_str = '_'.join([f'{k}_{v}' for k, v in hyperprior.items()])
            base_path = os.path.join(embeddings_folder, dataset_getter.dataset_name,
                                     f'{max_layers}_{epochs}_{unib}_{aggregation}_{hyperprior_str}_{alpha}_{gamma}_{sample_neighboring_macrostate}_{outer_k + 1}_{inner_k + 1}')

            train_out_emb = torch.cat([d.g_outs if unib else d.g_outs[:, :, :unigram_dim] for d in train_out], dim=0)
            val_out_emb = torch.cat([d.g_outs if unib else d.g_outs[:, :, :unigram_dim] for d in val_out], dim=0)
            test_out_emb = torch.cat([d.g_outs if unib else d.g_outs[:, :, :unigram_dim] for d in test_out], dim=0)

            torch.save(train_out_emb, base_path + '_train.torch')
            torch.save(val_out_emb, base_path + '_val.torch')
            torch.save(test_out_emb, base_path + '_test.torch')
            torch.save(torch.tensor(all_C, dtype=torch.int64), base_path + '_allC.torch')

        # CLEAR OUTPUTS
        for mode in ['train', 'validation', 'test']:
            shutil.rmtree(os.path.join(self.output_folder, mode), ignore_errors=True)

        tr_res = {LOSS: {'main_loss': torch.zeros(1)}, SCORE: {'main_score': torch.zeros(1)}}
        vl_res = {LOSS: {'main_loss': torch.zeros(1)}, SCORE: {'main_score': torch.zeros(1)}}
        return tr_res, vl_res

    def run_test(self, dataset_getter, logger):
        tr_res = {LOSS: {'main_loss': torch.zeros(1)}, SCORE: {'main_score': torch.zeros(1)}}
        vl_res = {LOSS: {'main_loss': torch.zeros(1)}, SCORE: {'main_score': torch.zeros(1)}}
        te_res = {LOSS: {'main_loss': torch.zeros(1)}, SCORE: {'main_score': torch.zeros(1)}}
        return tr_res, vl_res, te_res
