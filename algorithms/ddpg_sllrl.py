import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import os, math

from models.mlp import Actor, Critic, Hyper_Critic_Random, Hyper_Critic
from buffers import ReplayBuffer
import copy
from scipy.spatial import KDTree
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class DDPG_SLLRL(object):
    def __init__(self, state_dim, action_dim, max_action, lr=5e-4, gamma=0.99,
                 tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2, num_noise_samples=50, beta=0.001,
                 with_importance_sampling=0, device='cpu'):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        critic = Hyper_Critic

        self.critic = critic(state_dim, action_dim).to(device)
        self.critic_target = critic(state_dim, action_dim).to(device)
        self.copy_params(self.critic_target, self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=5e-5)

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0
        self.device = device
        self.action_dim = action_dim

        self.k = 1
        self.alpha = 2.5

        self.beta = beta
        self.num_noise_samples = num_noise_samples
        self.with_importance_sampling = with_importance_sampling
        self.device = device

        # Initialize the reference Gaussian
        self.kl_div_var = 0.15
        self.ref_gaussian = MultivariateNormal(torch.zeros(self.action_dim).to(self.device),
                                               torch.eye(self.action_dim).to(self.device) * self.kl_div_var)

    def load_from(self, agent):
        self.actor.load_state_dict(agent.actor.state_dict())
        self.actor_target.load_state_dict(agent.actor_target.state_dict())
        self.critic.load_state_dict(agent.critic.state_dict())
        self.critic_target.load_state_dict(agent.critic_target.state_dict())

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state)
        return action.cpu().data.numpy().flatten()

    def softmax_operator(self, q_vals, noise_pdf=None):
        max_q_vals = torch.max(q_vals, 1, keepdim=True).values
        norm_q_vals = q_vals - max_q_vals
        e_beta_normQ = torch.exp(self.beta * norm_q_vals)
        Q_mult_e = q_vals * e_beta_normQ

        numerators = Q_mult_e
        denominators = e_beta_normQ

        if self.with_importance_sampling:
            numerators /= noise_pdf
            denominators /= noise_pdf

        sum_numerators = torch.sum(numerators, 1)
        sum_denominators = torch.sum(denominators, 1)

        softmax_q_vals = sum_numerators / sum_denominators

        softmax_q_vals = torch.unsqueeze(softmax_q_vals, 1)
        return softmax_q_vals

    def calc_pdf(self, samples, mu=0):
        pdfs = 1 / (self.policy_noise * np.sqrt(2 * np.pi)) * torch.exp(
            - (samples - mu) ** 2 / (2 * self.policy_noise ** 2))
        pdf = torch.prod(pdfs, dim=2)
        return pdf

    def copy_params(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def reduce_lr(self, net_optimizer, lr=1e-5):
        for param_group in net_optimizer.param_groups:
            if param_group['lr'] != lr:
                print("### lr drop %s ###" % lr)
            param_group['lr'] = lr

    def compute_gradient(self, parameters):
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        if len(parameters) == 0:
            return torch.tensor(0.)
        device = parameters[0].grad.device
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0).to(device) for p in parameters]), 2.0)
        return total_norm.item()

    def disable_gradients(self):
        for p in self.critic.parameters():
            p.requires_grad = False

    def enable_gradients(self):
        for p in self.critic.parameters():
            p.requires_grad = True

    def update(self, replay_buffer, batch_size=32):
        self.total_it += 1
        # Sample replay buffer
        x, y, u, r, d = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(x).to(self.device)
        next_state = torch.FloatTensor(y).to(self.device)
        action = torch.FloatTensor(u).to(self.device)
        done = torch.FloatTensor(d).to(self.device)
        reward = torch.FloatTensor(r).to(self.device)

        states = x
        actions = u
        data = np.hstack([states, actions])
        kd_tree = KDTree(data)

        with torch.no_grad():
            noise = torch.randn(
                (action.shape[0], self.num_noise_samples, action.shape[1]),
                dtype=action.dtype, layout=action.layout, device=action.device
            )
            noise = noise * self.policy_noise
            noise_pdf = self.calc_pdf(noise) if self.with_importance_sampling else None

            noise1 = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise1).clamp(-self.max_action, self.max_action)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            # target_Q = torch.squeeze(target_Q, 2)
            target_Q = self.softmax_operator(target_Q, noise_pdf)
            target_Q = reward + ((1 - done) * self.gamma * target_Q).detach()

        ### get current Q estimate
        current_Q1, current_Q2 = self.critic(state, action)

        try:
            # Get current actions on the states of external experiences
            with torch.no_grad():
                current_action = self.actor(state)

            # Compute the difference batch
            diff_action_batch = action - current_action

            # Get the mean and covariance matrix for the
            mean = torch.mean(diff_action_batch, dim=0)
            cov = torch.mm(torch.transpose(diff_action_batch - mean, 0, 1), diff_action_batch - mean) / batch_size

            multivar_gaussian = MultivariateNormal(mean, cov)

            js_div = (kl_divergence(multivar_gaussian, self.ref_gaussian) + kl_divergence(self.ref_gaussian,
                                                                                          multivar_gaussian)) / 2

            js_div = torch.exp(-js_div)
            js_weights = js_div.item() * torch.ones_like(reward).to(self.device)
            if torch.isnan(js_weights).any():
                js_weights = torch.ones_like(reward).to(self.device)
            if torch.sum(js_weights) == 0:
                js_weights = torch.ones_like(reward).to(self.device)
        except:
            js_weights = torch.ones_like(reward).to(self.device)

        js_weights = F.softmax(js_weights, dim=0)

        # Compute critic loss
        critic_loss = torch.sum(js_weights * (F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)))

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = 0

        if self.total_it % self.policy_freq == 0:
            pi = self.actor(state)
            actor_loss = -(js_weights * self.critic.Q1(state, pi, None)).sum()

            key = torch.cat([state, pi], dim=1).detach().cpu().numpy()
            _, idx = kd_tree.query(key, k=[self.k], workers=-1)
            nearest_neighbour = (torch.tensor(data[idx][:, :, -self.action_dim:]).squeeze(dim=1).to(device)).type(
                pi.dtype)
            dc_loss = F.mse_loss(pi, nearest_neighbour.type(pi.dtype))

            combined_loss = actor_loss + dc_loss
            # combined_loss = actor_loss + 0.05*dc_loss
            combined_loss = combined_loss.to(torch.float32)

            # Optimize the actor
            self.actor_optimizer.zero_grad()

            self.disable_gradients()
            combined_loss.backward()
            self.enable_gradients()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss, critic_loss

    def compute_likelihood(self, transitions, sigma=0.25):
        # Sample replay buffer
        # x, y, u, r, d = replay_buffer.sample(len(replay_buffer.storage))
        S, Y, U, R, D = [], [], [], [], []
        for (s, y, u, r, d) in transitions:
            S.append(np.array(s, copy=False))
            Y.append(np.array(y, copy=False))
            U.append(np.array(u, copy=False))
            R.append(np.array(r, copy=False))
            D.append(np.array(d, copy=False))
        S = np.array(S)
        Y = np.array(Y)
        U = np.array(U)
        R = np.array(R).reshape(-1, 1)
        D = np.array(D).reshape(-1, 1)

        state = torch.FloatTensor(S).to(self.device)
        action = torch.FloatTensor(U).to(self.device)
        next_state = torch.FloatTensor(Y).to(self.device)
        done = torch.FloatTensor(D).to(self.device)
        reward = torch.FloatTensor(R).to(self.device)

        # Compute the target Q value
        target_Q = self.critic.Q1(next_state, self.actor(next_state), None)
        target_Q = reward + ((1 - done) * self.gamma * target_Q).detach()
        # Get current Q estimate
        current_Q = self.critic.Q1(state, action, None).detach()

        # Compute the mean likelihood
        a0 = torch.mul(target_Q - current_Q, target_Q - current_Q) / (sigma * sigma)
        a = -torch.clamp(a0, 0, 9) / 2

        p = a.exp() / (math.sqrt(2 * math.pi) * sigma)
        p_mean = p.log().mean().exp()
        return p_mean.cpu().numpy()

    def Bellman_residual(self, replay_buffer):
        # Sample replay buffer
        x, y, u, r, d = replay_buffer.sample(len(replay_buffer.storage))
        state = torch.FloatTensor(x).to(self.device)
        action = torch.FloatTensor(u).to(self.device)
        next_state = torch.FloatTensor(y).to(self.device)
        done = torch.FloatTensor(d).to(self.device)
        reward = torch.FloatTensor(r).to(self.device)

        # Compute the target Q value
        target_Q = self.critic.Q1(next_state, self.actor(next_state), None)
        target_Q = reward + ((1 - done) * self.gamma * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic.Q1(state, action, None).detach()

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)
        return critic_loss.cpu().data.numpy()
