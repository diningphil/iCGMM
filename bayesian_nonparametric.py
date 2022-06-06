import torch
from torch.distributions import Normal, Dirichlet, Gamma, Categorical


class BayesianDistribution(torch.nn.Module):
    EPS = 1e-18

    def __init__(self):
        super(BayesianDistribution, self).__init__()
        self.emission_distr = None

    def initialise_parameters(self):
        raise NotImplementedError('Must be implented in subclasses')

    def update_counts(self, data):
        raise NotImplementedError('Must be implented in subclasses')

    def update_parameters(self):
        raise NotImplementedError('Must be implented in subclasses')

    def get_data_log_likelihood(self, y_labels):
        raise NotImplementedError('Must be implented in subclasses')

    def __str__(self):
        return str(self.emission_distr)


class BNPCategorical(BayesianDistribution):

    def __init__(self, dim_target, alpha):
        """
        :param dim_target: dimension of output alphabet
        :param theta: the categorical emission distribution associated with a state
        """
        super().__init__()

        self.K = dim_target  # discrete output labels
        self.counts = torch.zeros(self.K)
        self.alpha = torch.ones(self.K) * alpha
        self.initialise_parameters()

    def _flatten_labels(self, labels):
        labels = torch.squeeze(labels)
        if len(labels.shape) > 1:
            # Compute discrete categories from one_hot_input
            labels_squeezed = labels.argmax(dim=1)
            return labels_squeezed
        return labels.long()

    def to(self, device):
        super().to(device)
        self.device = device
        self.counts.to(device)

    def initialise_parameters(self):
        d_prior = Dirichlet(self.alpha)
        self.emission_distr = Categorical(d_prior.sample())

    def update_counts(self, xu):
        self.counts[xu.argmax()] += 1

    def update_parameters(self):
        d_posterior = Dirichlet(self.counts + self.alpha)
        self.emission_distr = Categorical(d_posterior.sample())
        torch.zero_(self.counts)

    def get_data_log_likelihood(self, y_labels):
        y_labels_squeezed = self._flatten_labels(y_labels)
        # Returns the emission probability associated to each observable
        emission_obs_log = self.emission_distr.log_prob(y_labels_squeezed)
        return emission_obs_log


class BNPCategoricalBatch(BayesianDistribution):

    def __init__(self, dim_target, alpha):
        """
        :param dim_target: dimension of output alphabet
        :param theta: the value of the Dirichle prior
        """
        super(BNPCategoricalBatch, self).__init__()
        self.K = dim_target
        self.counts = torch.zeros(self.K)
        self.alpha = torch.ones(self.K) * alpha
        self.device = None
        self.initialise_parameters()

    def to(self, device):
        super().to(device)
        self.device = device
        self.counts.to(device)

    def initialise_parameters(self):
        d_prior = Dirichlet(self.alpha)
        self.emission_distr = Categorical(d_prior.sample())

    def update_counts(self, data):
        # we assume one-hot labels
        if len(data.shape) > 1:
            self.counts += data.sum(0)
        else:
            self.counts += data

    def get_data_log_likelihood(self, y_labels):
        # we assume one-hot labels
        # Returns the emission probability associated to each observable
        return self.emission_distr.log_prob(torch.argmax(y_labels, -1))

    def update_parameters(self):
        d_posterior = Dirichlet(self.counts + self.alpha)
        self.emission_distr = Categorical(d_posterior.sample())
        torch.zero_(self.counts)


class BNPGaussian(BayesianDistribution):

    def __init__(self, dim_target, mu0, lam0, a0, b0):
        """
        :param mean: mean
        :param var: variance
        """
        super(BNPGaussian, self).__init__()
        assert dim_target == 1, "only univariate case"
        self.mu0 = mu0
        self.lam0 = lam0
        self.a0 = a0
        self.b0 = b0
        self.initialise_parameters()
        self.device = None
        self.all_data_list = []

    def to(self, device):
        super().to(device)
        self.device = device

    def initialise_parameters(self):
        prec_prior = Gamma(self.a0, self.b0)
        precision = prec_prior.sample() + self.EPS

        mu_prior = Normal(self.mu0, 1. / torch.sqrt(self.lam0 * precision))
        mean = mu_prior.sample()
        self.emission_distr = Normal(mean, 1. / torch.sqrt(precision))

    def get_data_log_likelihood(self, y_labels):
        return self.emission_distr.log_prob(y_labels.squeeze())

    def update_counts(self, xu):
        self.all_data_list.append(xu)

    def update_parameters(self):
        # formulas taken from https://en.wikipedia.org/wiki/Normal-gamma_distribution
        if len(self.all_data_list) > 0:
            all_data = torch.cat(self.all_data_list, dim=0)
            sample_mean = torch.mean(all_data, dim=0)
            sample_var = torch.mean(torch.pow(all_data - sample_mean, 2), dim=0)
            n = all_data.shape[0]

            a0_post = self.a0 + n / 2.
            b0_post = self.b0 + 0.5 * (
                    n * sample_var + (self.lam0 * n * (sample_mean - self.mu0) ** 2) / (self.lam0 + n))
            mu0_post = (self.lam0 * self.mu0 + n * sample_mean) / (self.lam0 + n)
            lam0_post = self.lam0 + n

            prec_post = Gamma(a0_post, b0_post)
            precision = prec_post.sample() + self.EPS

            mu_post = Normal(mu0_post, 1. / torch.sqrt(lam0_post * precision))
            mean = mu_post.sample()

            self.emission_distr = Normal(mean, 1. / torch.sqrt(precision))
            # throw away data
            self.all_data_list = []
        else:
            # re-generate the parameters
            self.initialise_parameters()


class BNPGaussianBatch(BayesianDistribution):

    def __init__(self, dim_target, mu0, lam0, a0, b0):
        """
        :param mean: mean
        :param var: variance
        """
        super(BNPGaussianBatch, self).__init__()
        assert dim_target == 1, "only univariate case"
        self.mu0 = mu0
        self.lam0 = lam0
        self.a0 = a0
        self.b0 = b0
        self.initialise_parameters()
        self.device = None
        self.all_data_list = []

    def to(self, device):
        super().to(device)
        self.device = device

    def initialise_parameters(self):
        prec_prior = Gamma(self.a0, self.b0)
        precision = prec_prior.sample() + self.EPS

        mu_prior = Normal(self.mu0, 1. / torch.sqrt(self.lam0 * precision))
        mean = mu_prior.sample()
        self.emission_distr = Normal(mean, 1. / torch.sqrt(precision))

    def get_data_log_likelihood(self, y_labels):
        return self.emission_distr.log_prob(y_labels.squeeze())

    def update_counts(self, data):
        self.all_data_list.append(data)

    def update_parameters(self):
        # formulas taken from https://en.wikipedia.org/wiki/Normal-gamma_distribution
        if len(self.all_data_list) > 0:
            all_data = torch.cat(self.all_data_list, dim=0)
            sample_mean = torch.mean(all_data, dim=0)
            sample_var = torch.mean(torch.pow(all_data - sample_mean, 2), dim=0)
            n = all_data.shape[0]

            a0_post = self.a0 + n / 2.
            b0_post = self.b0 + 0.5 * (
                    n * sample_var + (self.lam0 * n * (sample_mean - self.mu0) ** 2) / (self.lam0 + n))
            mu0_post = (self.lam0 * self.mu0 + n * sample_mean) / (self.lam0 + n)
            lam0_post = self.lam0 + n

            prec_post = Gamma(a0_post, b0_post)
            precision = prec_post.sample() + self.EPS

            mu_post = Normal(mu0_post, 1. / torch.sqrt(lam0_post * precision))
            mean = mu_post.sample()

            self.emission_distr = Normal(mean, 1. / torch.sqrt(precision))
            # throw away data
            self.all_data_list = []
        else:
            # re-generate the parameters
            self.initialise_parameters()
