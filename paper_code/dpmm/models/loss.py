
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import linalg as LA
from torch.nn.functional import kl_div, softmax, log_softmax
from torch.distributions import MultivariateNormal

from scipy.stats import beta

from .mvnorm import multivariate_normal_cdf as Phi


class KLDivLoss(nn.Module):
    def __init__(self, temperature=0.2):
        super(KLDivLoss, self).__init__()

        self.temperature = temperature
    def forward(self, emb1, emb2):
        emb1 = softmax(emb1/self.temperature, dim=1).detach()
        emb2 = log_softmax(emb2/self.temperature, dim=1)
        loss_kldiv = kl_div(emb2, emb1, reduction='none')
        loss_kldiv = torch.sum(loss_kldiv, dim=1)
        loss_kldiv = torch.mean(loss_kldiv)
        return loss_kldiv


class RankingLoss(nn.Module):
    def __init__(self, neg_penalty=0.03):
        super(RankingLoss, self).__init__()

        self.neg_penalty = neg_penalty
    def forward(self, ranks, labels, class_ids_loaded, device):
        '''
        for each correct it should be higher then the absence 
        '''
        labels = labels[:, class_ids_loaded]
        ranks_loaded = ranks[:, class_ids_loaded]
        neg_labels = 1+(labels*-1)
        loss_rank = torch.zeros(1).to(device)
        for i in range(len(labels)):
            correct = ranks_loaded[i, labels[i]==1]
            wrong = ranks_loaded[i, neg_labels[i]==1]
            correct = correct.reshape((-1, 1)).repeat((1, len(wrong)))
            wrong = wrong.repeat(len(correct)).reshape(len(correct), -1)
            image_level_penalty = ((self.neg_penalty+wrong) - correct)
            image_level_penalty[image_level_penalty<0]=0
            loss_rank += image_level_penalty.sum()
        loss_rank /=len(labels)

        return loss_rank


class CosineLoss(nn.Module):
    
    def forward(self, cxr, ehr):
        a_norm = ehr / ehr.norm(dim=1)[:, None]
        b_norm = cxr / cxr.norm(dim=1)[:, None]
        loss = 1 - torch.mean(torch.diagonal(torch.mm(a_norm, b_norm.t()), 0))
        
        return loss


class DirichletProcessLoss(nn.Module):

    def __init__(self, dim=256, K=3, M=2, rho_scale=-4, eta=1):
        super(DirichletProcessLoss, self).__init__()
        """
        !!! One of the variants !!!
        """
        self.theta = nn.Parameter(torch.ones(1) * 1)

        self.pi = nn.Parameter(torch.ones([K * M]) / (K * M))
        self.phi = torch.ones(K * M) / (K * M)

        self.eta = eta
        self.gamma_1 = torch.ones(K * M)
        self.gamma_2 = torch.ones(K * M) * eta

        self.eta = eta

        self.mu_x = nn.Parameter(torch.zeros([K, dim]))
        self.mu_y = nn.Parameter(torch.zeros([K, dim]))
        self.log_cov_x = nn.Parameter(torch.ones([K, dim]) * rho_scale)
        self.log_cov_y = nn.Parameter(torch.ones([K, dim]) * rho_scale)

        self.K = K
        self.M = M
        self.n_mixture = K * M

    @property
    def cov_x(self):
        return torch.log1p(torch.exp(self.log_cov_x)).clamp(min=1e-15)

    @property
    def cov_y(self):
        return torch.log1p(torch.exp(self.log_cov_y)).clamp(min=1e-15)

    def _estimate_log_weights(self):
        digamma_sum = torch.digamma(
            self.gamma_1 + self.gamma_2
        ).cuda()
        digamma_a = torch.digamma(self.gamma_1).cuda()
        digamma_b = torch.digamma(self.gamma_2).cuda()

        return (
                digamma_a
                - digamma_sum
                + torch.hstack((
            torch.zeros(1, device=digamma_a.device),
            torch.cumsum(digamma_b - digamma_sum, 0)[:-1]
        ))
        )

    def _update_gamma(self):
        phi = self.phi

        phi_flipped = torch.flip(phi, dims=[1])
        cum_sum = torch.cumsum(phi_flipped, dim=1) - phi_flipped
        cum_sum = torch.flip(cum_sum, dims=[1])

        self.gamma_1 = 1 + phi.mean(0)
        self.gamma_2 = self.eta + cum_sum.mean(0)

    def forward(self, x, y):

        batch_size = x.shape[0]
        beta = self.sample_beta(batch_size)
        pi = self.mix_weights(beta)[:, :-1]

        cov_x = torch.log1p(torch.exp(self.log_cov_x)).clamp(min=1e-15)
        cov_y = torch.log1p(torch.exp(self.log_cov_y)).clamp(min=1e-15)

        # loss = torch.cat(c_list)
        u_log_pdf = [self.mvn_pdf(x, self.mu_x[k], cov_x[k]) for k in range(self.K)]
        v_log_pdf = [self.mvn_pdf(y, self.mu_y[k], cov_y[k]) for k in range(self.K)]

        u_entropy = [self.mvn_entropy(self.mu_x[k], cov_x[k]) for k in range(self.K)]
        v_entropy = [self.mvn_entropy(self.mu_y[k], cov_y[k]) for k in range(self.K)]

        assert not torch.isinf(torch.stack(u_log_pdf, dim=1)).any()
        assert not torch.isinf(torch.stack(v_log_pdf, dim=1)).any()

        loss = torch.stack(u_log_pdf + v_log_pdf, dim=1) + torch.stack(u_entropy + v_entropy, dim=0) + torch.log(pi.clamp(min=1e-15))

        self.phi = torch.softmax(loss, dim=-1).clamp(min=1e-15).detach()
        self._update_gamma()

        loss = torch.logsumexp(loss, -1).mean(0)

        assert loss.dim() == 0
        assert not torch.isnan(loss)

        return - loss  #  ELBO = negative of likelihood

    def rsample(self, n_samples=[0], return_x=False):
        """
        Sample (gradient-preserving) from Gaussian mixture
        Only for y
        """
        beta = self.sample_beta(n_samples[0])
        pi = self.mix_weights(beta)[:, :-1]

        cov_x = torch.log1p(torch.exp(self.log_cov_x)).clamp(min=1e-15)
        cov_y = torch.log1p(torch.exp(self.log_cov_y)).clamp(min=1e-15)

        # Using reparameterization tricks
        assert not torch.isnan(self.mu_x).any()
        assert not torch.isnan(cov_x).any()
        assert not torch.isnan(pi).any()
        assert not torch.isnan(self.mu_y).any()
        assert not torch.isnan(cov_y).any()

        x_samples = [MultivariateNormal(self.mu_x[k], scale_tril=torch.diag(cov_x[k])).rsample(sample_shape=n_samples).T * pi[:, k] for k in range(self.K)]
        y_samples = [MultivariateNormal(self.mu_y[k], scale_tril=torch.diag(cov_y[k])).rsample(sample_shape=n_samples).T * pi[:, self.K + k] for k in range(self.K)]

        x_samples = torch.stack(x_samples, dim=0).sum(dim=0).T.float()
        y_samples = torch.stack(y_samples, dim=0).sum(dim=0).T.float()
        assert not torch.isnan(x_samples).any()
        assert not torch.isnan(y_samples).any()

        assert not torch.isinf(x_samples).any()
        assert not torch.isinf(y_samples).any()

        if return_x:
            return x_samples


        return y_samples

    def infer(self, x):
        """
        Get logit

        return: Logits with length T
        """

        # TODO Change it to imputations

        beta = self.sample_beta(x.shape[0])
        pi = self.mix_weights(beta)[:, :-1]
        log_pdfs = self.get_log_prob(x)
        logits = torch.log(pi) + log_pdfs
        assert not torch.isnan(logits).any()

        return logits

    def sample_beta(self, size):
        a = self.gamma_1.detach().cpu().numpy()
        b = self.gamma_2.detach().cpu().numpy()

        samples = beta.rvs(a, b, size=(size, self.n_mixture))
        samples = torch.from_numpy(samples).cuda()

        return samples

    def mvn_pdf(self, x, mu, cov):
        """
        PDF of multivariate normal distribution
        """
        log_pdf = MultivariateNormal(mu, scale_tril=torch.diag(cov)).log_prob(x)

        return log_pdf

    def mvn_log_pdf(self, x):
        """
        PDF of multivariate normal distribution
        """

        return (-torch.log(torch.sqrt(2 * torch.pi))
                - torch.log(self.std_dev)
                - ((x - self.mu) ** 2) / (2 * self.std_dev ** 2)).sum(dim=-1)

    def mvn_entropy(self, mu, cov):
        """
        PDF of multivariate normal distribution
        """
        log_pdf = MultivariateNormal(mu, scale_tril=torch.diag(cov)).entropy()

        return log_pdf

    def mix_weights(self, beta):
        beta1m_cumprod = (1 - beta).cumprod(-1)
        pi = F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)
        return pi

    def update_phi(self):
        nk = self._estimate_nk(self.log_phi)
        self._update_gamma(nk)

    def set_log_phi(self, log_phi):
        self.log_phi = log_phi

class DirichletProcessLossSep(nn.Module):

    def __init__(self, dim=256, K=3, rho_scale=-4, eta=1):
        super(DirichletProcessLossSep, self).__init__()
        """
        !!! One of the variants !!!
        """
        self.theta = nn.Parameter(torch.ones(1) * 1)

        self.pi = nn.Parameter(torch.ones([K]) / (K))
        self.phi = torch.ones(K) / K

        self.eta = eta
        self.gamma_1 = torch.ones(K)
        self.gamma_2 = torch.ones(K) * eta

        self.eta = eta

        self.mu = nn.Parameter(torch.zeros([K, dim]))
        self.log_cov = nn.Parameter(torch.ones([K, dim]) * rho_scale)

        self.K = K
        self.n_mixture = K

    @property
    def cov_x(self):
        return torch.log1p(torch.exp(self.log_cov)).clamp(min=1e-15)

    def _estimate_log_weights(self):
        digamma_sum = torch.digamma(
            self.gamma_1 + self.gamma_2
        ).cuda()
        digamma_a = torch.digamma(self.gamma_1).cuda()
        digamma_b = torch.digamma(self.gamma_2).cuda()

        return (
                digamma_a
                - digamma_sum
                + torch.hstack((
            torch.zeros(1, device=digamma_a.device),
            torch.cumsum(digamma_b - digamma_sum, 0)[:-1]
        ))
        )

    def _update_gamma(self):
        phi = self.phi

        phi_flipped = torch.flip(phi, dims=[1])
        cum_sum = torch.cumsum(phi_flipped, dim=1) - phi_flipped
        cum_sum = torch.flip(cum_sum, dims=[1])

        self.gamma_1 = 1 + phi.mean(0)
        self.gamma_2 = self.eta + cum_sum.mean(0)

    def forward(self, x):

        batch_size = x.shape[0]
        beta = self.sample_beta(batch_size)
        pi = self.mix_weights(beta)[:, :-1]

        cov = torch.log1p(torch.exp(self.log_cov)).clamp(min=1e-15)

        # loss = torch.cat(c_list)
        u_log_pdf = [self.mvn_pdf(x, self.mu[k], cov[k]) for k in range(self.K)]
        u_entropy = [self.mvn_entropy(self.mu[k], cov[k]) for k in range(self.K)]

        loss = torch.stack(u_log_pdf, dim=1) + torch.stack(u_entropy, dim=0) + torch.log(pi.clamp(min=1e-15))

        self.phi = torch.softmax(loss, dim=-1).clamp(min=1e-15).detach()
        self._update_gamma()

        loss = torch.logsumexp(loss, -1).mean(0)

        return - loss  #  ELBO = negative of likelihood

    def rsample(self, n_samples=[0]):
        """
        Sample (gradient-preserving) from Gaussian mixture
        Only for y
        """
        beta = self.sample_beta(n_samples[0])
        pi = self.mix_weights(beta)[:, :-1]

        cov = torch.log1p(torch.exp(self.log_cov)).clamp(min=1e-15)

        # Using reparameterization tricks

        x_samples = [MultivariateNormal(self.mu[k], scale_tril=torch.diag(cov[k])).rsample(sample_shape=n_samples).T * pi[:, k] for k in range(self.K)]
        x_samples = torch.stack(x_samples, dim=0).sum(dim=0).T.float()

        return x_samples

    def infer(self, x):
        """
        Get logit

        return: Logits with length T
        """

        # TODO Change it to imputations

        beta = self.sample_beta(x.shape[0])
        pi = self.mix_weights(beta)[:, :-1]
        log_pdfs = self.get_log_prob(x)
        logits = torch.log(pi) + log_pdfs
        assert not torch.isnan(logits).any()

        return logits

    def sample_beta(self, size):
        a = self.gamma_1.detach().cpu().numpy()
        b = self.gamma_2.detach().cpu().numpy()

        samples = beta.rvs(a, b, size=(size, self.n_mixture))
        samples = torch.from_numpy(samples).cuda()

        return samples

    def mvn_pdf(self, x, mu, cov):
        """
        PDF of multivariate normal distribution
        """
        log_pdf = MultivariateNormal(mu, scale_tril=torch.diag(cov)).log_prob(x)

        return log_pdf

    def mvn_log_pdf(self, x):
        """
        PDF of multivariate normal distribution
        """

        return (-torch.log(torch.sqrt(2 * torch.pi))
                - torch.log(self.std_dev)
                - ((x - self.mu) ** 2) / (2 * self.std_dev ** 2)).sum(dim=-1)

    def mvn_entropy(self, mu, cov):
        """
        PDF of multivariate normal distribution
        """
        log_pdf = MultivariateNormal(mu, scale_tril=torch.diag(cov)).entropy()

        return log_pdf

    def mix_weights(self, beta):
        beta1m_cumprod = (1 - beta).cumprod(-1)
        pi = F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)
        return pi

    def update_phi(self):
        nk = self._estimate_nk(self.log_phi)
        self._update_gamma(nk)

    def set_log_phi(self, log_phi):
        self.log_phi = log_phi


class DirichletProcess3DLoss(nn.Module):

    def __init__(self, dim=256, K=3, M=3, rho_scale=-4, eta=1):
        super(DirichletProcess3DLoss, self).__init__()
        """
        !!! One of the variants !!!
        """
        self.theta = nn.Parameter(torch.ones(1) * 1)

        self.pi = nn.Parameter(torch.ones([K * M]) / (K * M))
        self.phi = torch.ones(K * M) / (K * M)

        self.eta = eta
        self.gamma_1 = torch.ones(K * M)
        self.gamma_2 = torch.ones(K * M) * eta

        self.eta = eta

        self.mu_x = nn.Parameter(torch.zeros([K, dim]))
        self.mu_y = nn.Parameter(torch.zeros([K, dim]))
        self.mu_z = nn.Parameter(torch.zeros([K, dim]))
        self.log_cov_x = nn.Parameter(torch.ones([K, dim]) * rho_scale)
        self.log_cov_y = nn.Parameter(torch.ones([K, dim]) * rho_scale)
        self.log_cov_z = nn.Parameter(torch.ones([K, dim]) * rho_scale)

        self.K = K
        self.M = M
        self.n_mixture = K * M

    def _update_gamma(self):
        phi = self.phi

        phi_flipped = torch.flip(phi, dims=[1])
        cum_sum = torch.cumsum(phi_flipped, dim=1) - phi_flipped
        cum_sum = torch.flip(cum_sum, dims=[1])

        self.gamma_1 = 1 + phi.mean(0)
        self.gamma_2 = self.eta + cum_sum.mean(0)

    def forward(self, x, y, z):

        batch_size = x.shape[0]
        beta = self.sample_beta(batch_size)
        pi = self.mix_weights(beta)[:, :-1]

        cov_x = torch.log1p(torch.exp(self.log_cov_x)).clamp(min=1e-15)
        cov_y = torch.log1p(torch.exp(self.log_cov_y)).clamp(min=1e-15)
        cov_z = torch.log1p(torch.exp(self.log_cov_z)).clamp(min=1e-15)

        # loss = torch.cat(c_list)
        u_log_pdf = [self.mvn_pdf(x, self.mu_x[k], cov_x[k]) for k in range(self.K)]
        v_log_pdf = [self.mvn_pdf(y, self.mu_y[k], cov_y[k]) for k in range(self.K)]
        w_log_pdf = [self.mvn_pdf(z, self.mu_z[k], cov_z[k]) for k in range(self.K)]

        u_entropy = [self.mvn_entropy(self.mu_x[k], cov_x[k]) for k in range(self.K)]
        v_entropy = [self.mvn_entropy(self.mu_y[k], cov_y[k]) for k in range(self.K)]
        w_entropy = [self.mvn_entropy(self.mu_z[k], cov_z[k]) for k in range(self.K)]

        assert not torch.isinf(torch.stack(u_log_pdf, dim=1)).any()
        assert not torch.isinf(torch.stack(v_log_pdf, dim=1)).any()
        assert not torch.isinf(torch.stack(w_log_pdf, dim=1)).any()

        loss = torch.stack(u_log_pdf + v_log_pdf + w_log_pdf, dim=1) + torch.stack(u_entropy + v_entropy + w_entropy, dim=0) + torch.log(pi.clamp(min=1e-15))

        self.phi = torch.softmax(loss, dim=-1).clamp(min=1e-15).detach()
        self._update_gamma()

        loss = torch.logsumexp(loss, -1).mean(0)

        assert loss.dim() == 0
        assert not torch.isnan(loss)

        return - loss  #  ELBO = negative of likelihood

    def sample_beta(self, size):
        a = self.gamma_1.detach().cpu().numpy()
        b = self.gamma_2.detach().cpu().numpy()

        samples = beta.rvs(a, b, size=(size, self.n_mixture))
        samples = torch.from_numpy(samples).cuda()

        return samples

    def mvn_pdf(self, x, mu, cov):
        """
        PDF of multivariate normal distribution
        """
        log_pdf = MultivariateNormal(mu, scale_tril=torch.diag(cov)).log_prob(x)

        return log_pdf

    def mvn_entropy(self, mu, cov):
        """
        PDF of multivariate normal distribution
        """
        log_pdf = MultivariateNormal(mu, scale_tril=torch.diag(cov)).entropy()

        return log_pdf

    def mix_weights(self, beta):
        beta1m_cumprod = (1 - beta).cumprod(-1)
        pi = F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)
        return pi


        

