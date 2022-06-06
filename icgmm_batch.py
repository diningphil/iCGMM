import torch
from pydgn.experiment.util import s2c
from torch.distributions import *
from torch.nn import ModuleList, Parameter
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_scatter import scatter_add, scatter_max

from util import compute_bigram, compute_unigram


def _my_scatter_nd(data, idx_tensor, shape):
    return torch.sparse_coo_tensor(idx_tensor, data, shape).coalesce().to_dense().to(data.device)


class iCGMMBatch(torch.nn.Module):

    def __init__(self, dim_node_features, dim_edge_features, dim_target,
                 readout_class, config):
        super(iCGMMBatch, self).__init__()
        self.device = None
        self.is_first_layer = config['depth'] == 1
        self.dim_node_features = dim_node_features
        self.dim_target = dim_target
        self.readout_class = readout_class
        self.depth = config['depth']
        self.training = False
        self.max_C = config['max_C']
        self.return_node_embeddings = False
        self.K = dim_node_features
        self.Y = dim_target

        # HYPER-PRIOR HERE IS GAMMA: see 8.8.2 BRML
        self.emission_distr_class = s2c(config['emission_distribution']['class'])
        self.emission_distr_prior_params = config['emission_distribution']['prior_params']

        # TODO: check this
        if 'alpha' in config and 'alpha_prior_params' in config:
            raise ValueError('Only one key between alpha or alpha_prior_params can be specified')
        if 'alpha' not in config and 'gamma_prior_params' not in config:
            raise ValueError('At least one key between alpha or alpha_prior_params must be specified')
        if 'alpha' in config:
            self.alpha = Parameter(torch.tensor(config['alpha'], dtype=torch.float32), requires_grad=False)
            self.alpha_prior_params = None
        elif 'alpha_prior_params' in config:
            self.alpha = None
            self.alpha_prior_params = {k: Parameter(torch.tensor(v, dtype=torch.float32), requires_grad=False)
                                       for k, v in config['alpha_prior_params'].items()}

        if 'gamma' in config and 'gamma_prior_params' in config:
            raise ValueError('Only one key between gamma or gamma_prior_params can be specified')
        if 'gamma' not in config and 'gamma_prior_params' not in config:
            raise ValueError('At least one key between gamma or gamma_prior_params must be specified')
        if 'gamma' in config:
            self.gamma = Parameter(torch.tensor(config['gamma'], dtype=torch.float32), requires_grad=False)
            self.gamma_prior_params = None
        elif 'gamma_prior_params' in config:
            self.gamma = None
            self.gamma_prior_params = {k: Parameter(torch.tensor(v, dtype=torch.float32), requires_grad=False)
                                       for k, v in config['gamma_prior_params'].items()}

        self.Ccurr = Parameter(torch.tensor(2, dtype=torch.int32),
                               requires_grad=False)  # start with 1/2 states and update during training
        self.A = 1  # fixed
        self.J = Parameter(torch.tensor(config['J'], dtype=torch.int32), requires_grad=False)
        # self.add_self_arc = config['self_arc'] if 'self_arc' in config else False
        self.sample_neighboring_macrostate = config['sample_neighboring_macrostate']
        self.use_continuous_states = True  # fixed
        self.unibigram = config['unibigram']
        self.aggregation = config['aggregation']

        # this varies over time, so it needs to be saved at the end of training
        # if not self.is_first_layer:
        # this is the emission/readout job
        # this varies over time, so it needs to be saved at the end of training
        self._init_alpha_gamma()
        self._init_beta()
        self._init_theta()
        self.init_permanent_accumulators()

    def _init_alpha_gamma(self):
        if self.alpha_prior_params is not None:
            alpha_prior_distr = Gamma(self.alpha_prior_params['a'],
                                      self.alpha_prior_params['b'])
            self.alpha = alpha_prior_distr.sample()

        if self.gamma_prior_params is not None:
            gamma_prior_distr = Gamma(self.gamma_prior_params['a'],
                                      self.gamma_prior_params['b'])
            self.gamma = gamma_prior_distr.sample()

    def _init_beta(self):
        # create all possible portions of the unitary stick in advance
        betas = torch.zeros(self.max_C)
        base_distrib = Beta(1, self.gamma)
        betas[0] = 1.
        for i in range(0, self.Ccurr):
            b = base_distrib.sample()
            tmp = betas[i].clone()
            betas[i] = b * tmp
            betas[i + 1] = (1 - b) * tmp
        # print(f'betas created: {betas}')
        assert torch.allclose(betas.sum(), torch.tensor(1.)), (betas, betas.sum())
        self.beta = torch.nn.Parameter(betas, requires_grad=False)

    def _init_theta(self):
        self.theta = ModuleList(
            [self.emission_distr_class(self.K, **self.emission_distr_prior_params) for _ in range(self.Ccurr + 1)])

    def init_permanent_accumulators(self, eval=False):
        """
        Accumulators that stay across Gibbs Sampling iterations
        (and therefore must be stored/loaded to/from a checkpoint)
        TODO tji and assignment matrices should be stored/loaded in checkpoints, but management is not straightforward
        """
        if not eval:
            self.njk = Parameter(torch.zeros(self.J, self.max_C), requires_grad=False)
            # 30 is a number chosen by us, it is increased dynamically
            # by doubling it when needed
            self.tji = Parameter(torch.zeros(self.J, self.max_C, 30), requires_grad=False)
            self.pos_next_table = torch.zeros(self.J, self.max_C, dtype=torch.long).to(self.device)

        self.macrostate_assignments_batch = []
        self.xji_z_assignment_batch = []
        self.xji_t_assignment_batch = []

    def to(self, device):
        super().to(device)
        self.device = device

        self.tji.to(device)
        self.njk.to(device)
        self.pos_next_table.to(device)
        for t in self.theta:
            t.to(device)

        return self

    def train(self):
        self.training = True
        # print('Training, initializing permanent accumulators')
        self.init_permanent_accumulators()
        for t in self.theta:
            t.train()

    def eval(self):
        self.training = False
        ################################
        #           WARNING            #
        ################################
        # TODO here I am assuming that whenever infer() is called in the training
        # engine, the eval() method is also called. This should be avoided somehow
        self.init_permanent_accumulators(eval=True)
        for t in self.theta:
            t.eval()

    def forward(self, data, id_batch):
        extra = None
        if not self.is_first_layer:
            data, extra = data[0], data[1]
        return self.sampling(data, id_batch, extra)

    def _add_new_state(self):
        # increment state counter
        self.Ccurr.data += 1
        # add new theta
        self.theta.append(self.emission_distr_class(self.K, **self.emission_distr_prior_params))

        # add new beta
        b = Beta(1, self.gamma).sample()
        beta_last = self.beta[self.Ccurr - 1].clone()
        self.beta[self.Ccurr - 1] = b * beta_last
        self.beta[self.Ccurr] = (1 - b) * beta_last

    def sampling(self, data, id_batch, extra=None):
        x, y, batch = data.x, data.y, data.batch

        prev_stats = None if self.is_first_layer else extra.vo_outs
        if prev_stats is not None:
            prev_stats.to(self.device)

        # THIS SHOULD IMPACT ONLY TRAINING
        # check if it is a new batch
        new_batch = False
        if id_batch >= len(self.xji_z_assignment_batch):
            new_batch = True

        # get restaurant assignments (j_batch)
        if self.is_first_layer:
            # in the first layer, only one restaurant
            j_batch = torch.zeros(x.shape[0], dtype=torch.long).to(self.device)
        else:
            if new_batch:
                # NEVER USED, IT DEGRADES PERFORMANCES
                if self.sample_neighboring_macrostate:
                    j_batch = Categorical(prev_stats[:, 0, :-1]).sample()
                else:
                    j_batch = torch.argmax(prev_stats[:, 0, :-1], 1)
                self.macrostate_assignments_batch.append(j_batch)
            else:
                # get macrostate assignment already computed
                j_batch = self.macrostate_assignments_batch[id_batch]

        # --------------------------- SAMPLE Z --------------------------- #
        # it is a sequential process that updates nk or njk
        # HDP behavior

        # calcola data likelihood per ogni stato (+1 nuovo stato)
        f_X_theta_log = torch.empty(x.shape[0], self.Ccurr + 1)
        for k in range(self.Ccurr + 1):
            f_X_theta_log[:, k] = self.theta[k].get_data_log_likelihood(x)

        # quantità che dipendo dal batch e che sono riusate più volte
        n_batch = x.shape[0]
        all_idx = torch.arange(0, n_batch).to(self.device)
        ones_batch = torch.ones(n_batch).to(self.device)

        if not new_batch:
            # batch data already seen
            xji_z_batch = self.xji_z_assignment_batch[id_batch]
            xji_t_batch = self.xji_t_assignment_batch[id_batch]
            z_idx_batch = torch.stack((j_batch, xji_z_batch), dim=1).t().to(self.device)
            t_idx_batch = torch.stack((j_batch, xji_z_batch, xji_t_batch), dim=1).t().to(self.device)

            if self.training:
                # Remove ji assignment from the total count
                self.njk -= _my_scatter_nd(ones_batch, z_idx_batch, self.njk.shape)
                self.tji -= _my_scatter_nd(ones_batch, t_idx_batch, self.tji.shape)

        # compute p_zu
        unnorm_log_p_zu = (self.alpha * self.beta[:self.Ccurr + 1] +
                           self.njk[j_batch, :self.Ccurr + 1]).log() + f_X_theta_log

        # remove inf
        inf_mask = torch.isinf(unnorm_log_p_zu)
        if torch.any(inf_mask):
            sign_p_zu = torch.sign(unnorm_log_p_zu)
            unnorm_log_p_zu[inf_mask] = sign_p_zu[inf_mask] * torch.full_like(unnorm_log_p_zu, 10 ** 18)[inf_mask]

        # log-sum epx trick
        max_log_v = torch.max(unnorm_log_p_zu, dim=1, keepdim=True).values
        log_zu = (unnorm_log_p_zu - max_log_v).exp().sum(1, keepdim=True).log() + max_log_v
        p_zu = (unnorm_log_p_zu - log_zu).exp()

        if self.Ccurr == self.max_C - 1:  # deal with limit case
            zu = Categorical(p_zu[:, :-1]).sample()
        else:
            zu = Categorical(p_zu).sample()

        if self.training:
            new_z_idx_batch = torch.stack((j_batch, zu), dim=1).t().to(self.device)

            if torch.any(zu == self.Ccurr):
                self._add_new_state()

            ######################### TABLE ASSIGNMENT ##########################
            unnorm_p_tji = self.tji[j_batch, zu, :]
            unnorm_p_tji[all_idx, self.pos_next_table[j_batch, zu]] = self.alpha * self.beta[zu]
            tu = Categorical(unnorm_p_tji).sample()

            i_with_new_table = (tu == self.pos_next_table[j_batch, zu]).long()
            self.pos_next_table += (_my_scatter_nd(i_with_new_table, new_z_idx_batch, self.pos_next_table.shape) >= 1)

            if torch.any(self.pos_next_table >= self.tji.size(2)):
                # double tji
                # aggiungi 30 alla volta?
                self.tji.data = torch.cat([self.tji, torch.zeros_like(self.tji)], dim=2).to(self.device)

            new_t_idx_batch = torch.stack((j_batch, zu, tu), dim=1).t().to(self.device)
            self.tji += _my_scatter_nd(ones_batch, new_t_idx_batch, self.tji.shape)
            ##################################################################

            self.njk += _my_scatter_nd(ones_batch, new_z_idx_batch, self.njk.shape)

            if new_batch:
                self.xji_z_assignment_batch.append(zu)
                self.xji_t_assignment_batch.append(tu)
            else:
                self.xji_z_assignment_batch[id_batch] = zu
                self.xji_t_assignment_batch[id_batch] = tu

            # updata theta stats
            for k in range(self.Ccurr):
                mask = zu == k
                if torch.any(mask):
                    self.theta[k].update_counts(x[mask])

        complete_log_likelihood = (p_zu * (f_X_theta_log)).sum(1).sum()
        complete_log_likelihood_2 = f_X_theta_log[all_idx, zu].sum()
        # complete_log_likelihood_3 = p_zu[all_idx, zu].log().sum()

        if self.return_node_embeddings:
            posterior_matrix = p_zu[:, :-1]

            statistics_batch = self._compute_statistics(posterior_matrix, data, self.device)

            node_unigram = compute_unigram(posterior_matrix, self.use_continuous_states)
            graph_unigram = self._get_aggregation_fun()(node_unigram, batch)

            if self.unibigram:
                node_bigram = compute_bigram(posterior_matrix.float(), data.edge_index, batch,
                                              self.use_continuous_states)
                graph_bigram = self._get_aggregation_fun()(node_bigram, batch)

                node_embeddings_batch = torch.cat((node_unigram, node_bigram), dim=1)
                graph_embeddings_batch = torch.cat((graph_unigram, graph_bigram), dim=1)
            else:
                node_embeddings_batch = node_unigram
                graph_embeddings_batch = graph_unigram

            embeddings = (None, None, graph_embeddings_batch, statistics_batch, None, None)
        else:
            embeddings = None

        num_nodes = x.shape[0]
        return None, embeddings, complete_log_likelihood/num_nodes, complete_log_likelihood_2/num_nodes  # , complete_log_likelihood_3

    def _update_alpha_gamma(self):
        nodes_for_each_group = self.njk.sum(dim=1)
        tot_number_of_tables = (self.tji > 0).sum()

        if self.alpha_prior_params is not None:
            # sampling schema for alpha (from "Hierarchical Dirichlet Processes" by Jordan)
            w_aux = torch.zeros(self.J)
            s_aux = torch.zeros(self.J)
            for j in range(self.J):
                # Eq. (48)
                w_aux_distr = Beta(self.alpha + 1, nodes_for_each_group[j])
                w_aux[j] = w_aux_distr.sample()
                # Eq. (49)
                s_aux_distr = Bernoulli(nodes_for_each_group[j] / (nodes_for_each_group[j] + self.alpha))
                s_aux[j] = s_aux_distr.sample()

            # Eq. (47)
            alpha_distr = Gamma(self.alpha_prior_params['a'] + tot_number_of_tables - torch.sum(s_aux),
                                self.alpha_prior_params['b'] - torch.sum(torch.log(w_aux)))
            self.alpha = alpha_distr.sample()

        if self.gamma_prior_params is not None:
            # sampling schema for gamma (from "THE STICKY HDP-HMM" by Fox et al)

            # Eq. (D.8)
            eta_distr = Beta(self.gamma + 1,
                             tot_number_of_tables)
            eta = eta_distr.sample()

            zeta_distr = Bernoulli(tot_number_of_tables / (tot_number_of_tables + self.gamma))
            zeta = zeta_distr.sample()

            gamma_distr = Gamma(self.gamma_prior_params['a'] + self.Ccurr - zeta,
                                self.gamma_prior_params['b'] - torch.log(eta))
            self.gamma = gamma_distr.sample()

    def update(self):
        # Check whether we need to add/remove states
        # states_to_remove = [k for k in range(self.Ccurr) if self.njk.sum(dim=0)[k] == 0]
        count_item_per_state = self.njk.sum(dim=0)
        states_to_keep = count_item_per_state > 0
        states_to_remove = count_item_per_state == 0
        n_states_to_remove = torch.sum(states_to_remove[:self.Ccurr])

        if n_states_to_remove > 0:
            self.Ccurr.data -= n_states_to_remove
            assert self.Ccurr == torch.sum(states_to_keep)

            self.njk.data[:, :self.Ccurr] = self.njk.data[:, states_to_keep]
            self.njk.data[:, self.Ccurr:] = 0

            self.pos_next_table.data[:, :self.Ccurr] = self.pos_next_table.data[:, states_to_keep]
            self.pos_next_table.data[:, self.Ccurr:] = 0

            '''
            it gets re-sampled later
            remaining_stick = self.beta.data[states_to_remove].sum()
            self.beta.data[:self.Ccurr] = self.beta.data[states_to_keep]
            self.beta.data[self.Ccurr+1] += remaining_stick.sum()
            '''

            self.tji[:, :self.Ccurr, :] = self.tji[:, states_to_keep, :]
            self.tji[:, self.Ccurr:, :] = 0

            old_Ccur = self.Ccurr.data + n_states_to_remove
            new_mapping = -torch.ones(old_Ccur, dtype=torch.long)
            new_mapping[states_to_keep[:old_Ccur]] = torch.arange(0, self.Ccurr).to(self.device)

            for i in range(len(self.xji_z_assignment_batch)):
                self.xji_z_assignment_batch[i] = new_mapping[self.xji_z_assignment_batch[i]]
                # assert(torch.all(self.xji_z_assignment_batch[i]>=0))

            for k in range(old_Ccur - 1, -1, -1):
                if states_to_remove[k]:
                    del self.theta[k]

        # Create backup tables to restore at eval time
        # self.backup_njk = self.njk.data.clone()

        '''

        TODO REMOVE UNOCCUPIED TABLES! EASIER THAN ABOVE, NOT CONNECTED
        TO ANY PARAMS, BUT BEWARE OF SHIFTING TABLE ASSIGNMENTS INDICES!

        '''
        beta_posterior_params = (self.tji > 0).sum(dim=2).sum(dim=0).float()
        beta_posterior_params[self.Ccurr] = self.gamma
        beta_posterior_params[beta_posterior_params == 0.] = 1e-16
        self.beta.data = Dirichlet(beta_posterior_params).sample()

        # update thetas (possibly including new state istantiated in the E-step)
        # (possibly excluding the "new new state")
        for k in range(self.Ccurr + 1):  # +1 to update also the "new" state
            self.theta[k].update_parameters()

        self._update_alpha_gamma()

    def stopping_criterion(self, depth, max_layers, train_loss, train_score, val_loss, val_score,
                           dict_per_layer, layer_config, logger=None):
        return depth == max_layers

    def _compute_statistics(self, posteriors, data, device):

        statistics = torch.full((posteriors.shape[0], self.A, posteriors.shape[1] + 1), 0., dtype=torch.float32).to(
            device)
        srcs, dsts = data.edge_index

        assert self.A == 1

        sparse_adj_matr = torch.sparse_coo_tensor(data.edge_index, \
                                                  torch.ones(data.edge_index.shape[1], dtype=posteriors.dtype).to(
                                                      device), \
                                                  torch.Size([posteriors.shape[0],
                                                              posteriors.shape[0]])).to(device).transpose(0, 1)
        statistics[:, 0, :-1] = torch.sparse.mm(sparse_adj_matr, posteriors)

        # Deal with nodes with degree 0: add a single fake neighbor with uniform posterior
        degrees = statistics[:, :, :-1].sum(dim=[1, 2]).floor()
        statistics[degrees == 0., :, :] = 1. / self.Ccurr.float()

        return statistics

    def _compute_sizes(self, batch, device):
        return scatter_add(torch.ones(len(batch), dtype=torch.int).to(device), batch)

    def _compute_max_ariety(self, degrees, batch):
        return scatter_max(degrees, batch)

    def _get_aggregation_fun(self):
        if self.aggregation == 'mean':
            aggregate = global_mean_pool
        elif self.aggregation == 'sum':
            aggregate = global_add_pool
        return aggregate
