import torch
from pydgn.experiment.util import s2c
from torch.distributions import *
from torch.nn import ModuleList, Parameter
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_scatter import scatter_add, scatter_max

from util import compute_unigram, compute_bigram


class iCGMMSeq(torch.nn.Module):

    def __init__(self, dim_node_features, dim_edge_features, dim_target,
                 readout_class, config):
        super(iCGMMSeq, self).__init__()
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

        self._init_alpha_gamma()
        self.beta = self._init_beta()
        self.theta = self._init_theta()
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
        betas = torch.zeros(self.Ccurr + 1)
        base_distrib = Beta(1, self.gamma)

        # stick breaking process to inizialize beta
        betas[0] = 1.
        for i in range(0, self.Ccurr):
            b = base_distrib.sample()
            tmp = betas[i].clone()
            betas[i] = b * tmp
            betas[i + 1] = (1 - b) * tmp
        assert torch.allclose(betas.sum(), torch.tensor(1.)), (betas, betas.sum())
        return torch.nn.Parameter(betas, requires_grad=False)

    def _init_theta(self):
        assert self.Ccurr == 2
        theta = ModuleList(
            [self.emission_distr_class(self.K, **self.emission_distr_prior_params) for _ in range(self.Ccurr + 1)])
        return theta

    def init_permanent_accumulators(self):
        """
        Accumulators that stay across Gibbs Sampling iterations
        (and therefore must be stored/loaded to/from a checkpoint)

        TODO tji and assignment matrices should be stored/loaded in checkpoints, but management is not straightforward

        """
        self.njk = Parameter(torch.zeros(self.J, self.Ccurr + 1), requires_grad=False)
        # counts the table assignments for each group j (restaurant) and mixture component k (dish)
        self.tji = [[[0] for _ in range(self.Ccurr)] for _ in range(self.J)]
        self.mjk = torch.zeros(self.J, self.Ccurr).to(self.device)
        self.macrostate_assignments = None

        self.xji_z_assignment = None
        self.xji_t_assignment = None
        self.first_run = True

        # Do not delete this!
        if self.device:  # set by to() method
            self.to(self.device)

    def _add_beta(self):
        # stick breaking process for new state
        betas = self.beta
        base_distrib = Beta(1, self.gamma)
        b = base_distrib.sample()
        beta_last = betas[-1].clone()
        betas[-1] = b * beta_last
        beta_new = (1 - b) * beta_last
        betas = torch.cat((betas, beta_new.unsqueeze(0)), dim=0)
        self.beta.data = betas
        return self.beta

    def _add_new_state(self):
        # increment state counter
        self.Ccurr.data += 1
        # add new theta
        self.theta.append(self.emission_distr_class(self.K, **self.emission_distr_prior_params))

    def to(self, device):
        super().to(device)
        self.device = device

        self.njk.to(device)
        self.mjk.to(device)

        for t in self.theta:
            t.to(device)

        return self

    def train(self):
        self.training = True
        for t in self.theta:
            t.train()
        # print('Training, initializing permanent accumulators')
        self.init_permanent_accumulators()

    def eval(self):
        self.training = False
        ################################
        #           WARNING            #
        ################################
        # TODO here I am assuming that whenever infer() is called in the training
        # engine, the eval() method is also called. This should be avoided somehow
        self.first_run = True
        for t in self.theta:
            t.eval()

    def forward(self, data, id_batch):
        extra = None
        if not self.is_first_layer:
            data, extra = data[0], data[1]
        return self.sampling(data, extra)

    def sampling(self, data, extra=None):
        x, y, batch = data.x, data.y, data.batch

        prev_stats = None if self.is_first_layer else extra.vo_outs
        if prev_stats is not None:
            prev_stats.to(self.device)

        # assigno to x to groups
        if self.macrostate_assignments is None:  # self.first_run:
            if self.is_first_layer:
                # all x to the same group
                self.macrostate_assignments = torch.zeros(x.shape[0], dtype=torch.int64).to(x.device)
            else:
                # sample neighbors macrostates
                neighbors_macrostates = prev_stats[:, 0, :-1]  # assume only 1 previous layer and no edge states
                self.macrostate_assignments = Categorical(neighbors_macrostates).sample()

        # compute f(x_u|theta_k) for all k (shape ?x(Ccurr+1))
        f_X_theta_log = None
        for k in range(self.Ccurr + 1):
            log_emission = self.theta[k].get_data_log_likelihood(x)  # ?
            if f_X_theta_log is None:
                f_X_theta_log = log_emission.unsqueeze(1)
            else:
                f_X_theta_log = torch.cat((f_X_theta_log, log_emission.unsqueeze(1)), dim=1)

        if self.training:
            if self.xji_z_assignment is None:
                ################################
                #           WARNING            #
                ################################
                # WE ASSUME FULL BATCH HERE!
                # TODO: ALSO, THIS SHOULD BE A PARAMETER TO BE STORED/LOADED!!
                # self.xji_z_assignment = (torch.zeros(x.shape[0], dtype=torch.int64) - 1).to(x.device)
                self.xji_z_assignment = torch.randint(self.Ccurr.item(), [x.shape[0]], dtype=torch.int64).to(x.device)
                idx = torch.stack((self.macrostate_assignments, self.xji_z_assignment), axis=0)
                vals = torch.ones(idx.shape[1])
                self.njk += torch.sparse_coo_tensor(idx, vals, size=self.njk.shape).coalesce().to_dense()

                self.xji_t_assignment = (torch.zeros(x.shape[0], dtype=torch.int64) - 1).to(x.device)

        # --------------------------- SAMPLE Z --------------------------- #

        posterior_matrix = torch.zeros(x.shape[0], self.Ccurr + 1).to(x.device)
        complete_log_likelihood_3 = torch.zeros(1).to(x.device)
        complete_log_likelihood_2 = torch.zeros(1).to(x.device)

        for u in range(x.shape[0]):

            j = self.macrostate_assignments[u]

            if self.training:
                # Remove ji assignment from the total count
                if self.xji_z_assignment[u] != -1:
                    old_p_zu = self.xji_z_assignment[u]
                    self.njk[j, old_p_zu] -= 1
                else:
                    old_p_zu = None

            unnorm_log_p_zu = (self.alpha * self.beta + self.njk[j]).log() + f_X_theta_log[u]
            # log-sum epx trick
            max_log_v = torch.max(unnorm_log_p_zu, dim=0, keepdim=True).values
            log_zu = (unnorm_log_p_zu - max_log_v).exp().sum(0, keepdim=True).log() + max_log_v
            p_zu = (unnorm_log_p_zu - log_zu).exp()

            # populate posterior matrix
            posterior_matrix[u] += p_zu

            if self.Ccurr == self.max_C:  # deal with limit case
                zu = Categorical(p_zu[:-1]).sample()
            else:
                zu = Categorical(p_zu).sample()

            complete_log_likelihood_2 += f_X_theta_log[u, zu]
            complete_log_likelihood_3 += p_zu[zu].log()

            if self.training:
                if zu == (self.Ccurr):
                    # add a state
                    self._add_new_state()
                    log_emission = self.theta[-1].get_data_log_likelihood(x)
                    f_X_theta_log = torch.cat((f_X_theta_log, log_emission.unsqueeze(1)), dim=1)

                    posterior_matrix = torch.cat((posterior_matrix, torch.zeros(x.shape[0], 1).to(x.device)), dim=1)
                    self.njk.data = torch.cat((self.njk.data, torch.zeros(self.J, 1).to(x.device)), dim=1)

                    for restaurant in range(self.J):  # do not use j as index here
                        self.tji[restaurant].append([])  # append new list of tables associated with new state

                    self.mjk = torch.cat((self.mjk, torch.zeros(self.J, 1).to(x.device)), dim=1)
                    self._add_beta()  # append new beta at the end

                # ONLY ONE DISH PER TABLE!
                # Knowing z, we can compute the conditional table assignments

                # assert len(self.tji) == self.J, (len(self.tji), self.J)
                # for debug_j in range(len(self.tji)):
                #    assert len(self.tji[debug_j]) == self.Ccurr.item(), (len(self.tji[debug_j]), self.Ccurr.item())

                if len(self.tji[j][zu]) == 0:  # no tables yet
                    self.tji[j][zu].append(1)  # create table with one customer
                    self.mjk[j, zu] += 1  # new table, update count
                    # assign to table
                    self.xji_t_assignment[u] = 0
                else:  # sample table

                    # Remove ji table assignment from the total count
                    if self.xji_t_assignment[u] != -1:
                        assert old_p_zu is not None

                        # assert len(self.tji[j]) > old_p_zu, (len(self.tji[j]), old_p_zu)
                        # assert len(self.tji[j]) == self.Ccurr, (len(self.tji[j]), self.Ccurr)
                        # assert len(self.tji[j][old_p_zu]) > self.xji_t_assignment[u], (len(self.tji[j][old_p_zu]), self.xji_t_assignment[u])
                        self.tji[j][old_p_zu][self.xji_t_assignment[u]] -= 1

                        if self.tji[j][old_p_zu][self.xji_t_assignment[u]] == 0:
                            # the table has become empty, decrease mjk
                            self.mjk[j, old_p_zu] -= 1

                            '''
                            TODO MAKE THIS WORK, but is not necessary to the functioning of iCGMM

                            # remove the table until we need a new one
                            del self.tji[j][old_p_zu][self.xji_t_assignment[u]]

                            # shift table idx only of those nodes with same restaurant and same dish
                            nodes_zu_j  = torch.logical_and((self.xji_z_assignment == old_p_zu), (self.macrostate_assignments == j))
                            index = self.xji_t_assignment[u]
                            self.xji_t_assignment[nodes_zu_j][self.xji_t_assignment[nodes_zu_j] > index] -= 1

                            '''
                    # use the sampled zu here
                    unnorm_p_tji = torch.tensor(self.tji[j][zu] + [self.alpha * self.beta[zu]]).to(x.device)
                    t = Categorical(unnorm_p_tji).sample()
                    if t == len(self.tji[j][zu]):  # add new table to group j for mixture component k
                        self.tji[j][zu].append(1)
                        self.mjk[j, zu] += 1  # new table, update count
                    else:
                        # Add ji table assignment to the total count
                        self.tji[j][zu][t] += 1

                    # assign to table
                    self.xji_t_assignment[u] = t

                self.theta[zu].update_counts(x[u])

                self.njk[j, zu] += 1
                self.xji_z_assignment[u] = zu

        # ASSUMES FULL BATCH
        self.first_run = False

        # leave this here (do not be tempted to move it above)
        complete_log_likelihood = (posterior_matrix * (f_X_theta_log)).sum(1).sum()

        # --------------------- HANDLE new/dead states ------------------- #

        if self.return_node_embeddings:
            posterior_matrix = posterior_matrix[:, :-1]

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
        return None, embeddings, complete_log_likelihood/num_nodes, \
               complete_log_likelihood_2/num_nodes, \
               complete_log_likelihood_3/num_nodes, num_nodes

    def _update_alpha_gamma(self):
        nodes_for_each_group = self.njk.sum(dim=1)
        tot_number_of_tables = self.mjk.sum()

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
        states_to_remove = [k for k in range(self.Ccurr) if self.njk.sum(dim=0)[k] == 0]
        # Q: How is this handled in other libraries?
        for index in sorted(states_to_remove, reverse=True):
            del self.theta[index]
            self.njk.data = torch.cat((self.njk.data[:, :index], self.njk.data[:, index + 1:]), dim=1)

            # reduce index value of state assignments that come after the
            # removed index (remember, we are iterating in reverse order)
            self.xji_z_assignment[self.xji_z_assignment > index] -= 1

            for j in range(0, self.J):
                del self.tji[j][index]

            self.mjk.data = torch.cat((self.mjk.data[:, :index],
                                       self.mjk.data[:, index + 1:]),
                                      dim=1)

            self.Ccurr.data -= 1

        unnorm_p_beta = torch.cat((self.mjk.sum(dim=0),
                                   torch.tensor([self.gamma])))
        p_beta = unnorm_p_beta / unnorm_p_beta.sum()

        self.beta.data = Dirichlet(p_beta).sample()

        # update thetas, including resampling the categorical for the new state
        for k in range(self.Ccurr + 1):
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
