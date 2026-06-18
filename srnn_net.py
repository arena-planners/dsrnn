"""Standalone inference reimplementation of the CrowdNav_DSRNN SRNN policy.

This reconstructs *only* the test-time forward path of the upstream policy
(``pytorchBaselines/a2c_ppo_acktr/{model,srnn_model,distributions,utils}.py``)
so the committed checkpoint loads with ``strict=True`` outside the upstream
``crowd_sim`` simulator and without the openai-baselines / KFAC training stack.

Module and parameter names match upstream exactly so the saved ``state_dict``
keys (``base.*``, ``dist.*``) line up one-for-one.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# --- upstream pytorchBaselines/a2c_ppo_acktr/utils.py -----------------------
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class AddBias(nn.Module):
    def __init__(self, bias):
        super().__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)
        return x + bias


# --- upstream pytorchBaselines/a2c_ppo_acktr/distributions.py ---------------
class FixedNormal(torch.distributions.Normal):
    def mode(self):
        return self.mean


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)
        zeros = torch.zeros(action_mean.size())
        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())


# --- upstream pytorchBaselines/a2c_ppo_acktr/srnn_model.py ------------------
def reshapeT(T, seq_length, nenv):
    shape = T.size()[1:]
    return T.unsqueeze(0).reshape((seq_length, nenv, *shape))


class RNNBase(nn.Module):
    def __init__(self, config, edge):
        super().__init__()
        self.config = config
        if edge:
            self.gru = nn.GRU(
                config.SRNN.human_human_edge_embedding_size,
                config.SRNN.human_human_edge_rnn_size,
            )
        else:
            self.gru = nn.GRU(
                config.SRNN.human_node_embedding_size * 2,
                config.SRNN.human_node_rnn_size,
            )

    def _forward_gru(self, x, hxs, masks):
        # acting model only (input shape[0] == hidden state shape[0])
        seq_len, nenv, agent_num, _ = x.size()
        x = x.view(seq_len, nenv * agent_num, -1)
        hxs_times_masks = hxs * (masks.view(seq_len, nenv, 1, 1))
        hxs_times_masks = hxs_times_masks.view(seq_len, nenv * agent_num, -1)
        x, hxs = self.gru(x, hxs_times_masks)
        x = x.view(seq_len, nenv, agent_num, -1)
        hxs = hxs.view(seq_len, nenv, agent_num, -1)
        return x, hxs


class HumanNodeRNN(RNNBase):
    def __init__(self, config):
        super().__init__(config, edge=False)
        self.config = config
        self.rnn_size = config.SRNN.human_node_rnn_size
        self.output_size = config.SRNN.human_node_output_size
        self.embedding_size = config.SRNN.human_node_embedding_size
        self.input_size = config.SRNN.human_node_input_size
        self.edge_rnn_size = config.SRNN.human_human_edge_rnn_size
        self.encoder_linear = nn.Linear(self.input_size, self.embedding_size)
        self.relu = nn.ReLU()
        self.edge_embed = nn.Linear(self.edge_rnn_size, self.embedding_size)
        self.edge_attention_embed = nn.Linear(self.edge_rnn_size * 2, self.embedding_size)
        self.output_linear = nn.Linear(self.rnn_size, self.output_size)

    def forward(self, pos, h_temporal, h_spatial_other, h, masks):
        encoded_input = self.relu(self.encoder_linear(pos))
        h_edges = torch.cat((h_temporal, h_spatial_other), -1)
        h_edges_embedded = self.relu(self.edge_attention_embed(h_edges))
        concat_encoded = torch.cat((encoded_input, h_edges_embedded), -1)
        x, h_new = self._forward_gru(concat_encoded, h, masks)
        outputs = self.output_linear(x)
        return outputs, h_new


class HumanHumanEdgeRNN(RNNBase):
    def __init__(self, config):
        super().__init__(config, edge=True)
        self.config = config
        self.rnn_size = config.SRNN.human_human_edge_rnn_size
        self.embedding_size = config.SRNN.human_human_edge_embedding_size
        self.input_size = config.SRNN.human_human_edge_input_size
        self.encoder_linear = nn.Linear(self.input_size, self.embedding_size)
        self.relu = nn.ReLU()

    def forward(self, inp, h, masks):
        encoded_input = self.relu(self.encoder_linear(inp))
        x, h_new = self._forward_gru(encoded_input, h, masks)
        return x, h_new


class EdgeAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.human_human_edge_rnn_size = config.SRNN.human_human_edge_rnn_size
        self.human_node_rnn_size = config.SRNN.human_node_rnn_size
        self.attention_size = config.SRNN.attention_size
        self.temporal_edge_layer = nn.ModuleList()
        self.spatial_edge_layer = nn.ModuleList()
        self.temporal_edge_layer.append(nn.Linear(self.human_human_edge_rnn_size, self.attention_size))
        self.spatial_edge_layer.append(nn.Linear(self.human_human_edge_rnn_size, self.attention_size))
        self.agent_num = 1
        self.num_attention_head = 1

    def att_func(self, temporal_embed, spatial_embed, h_spatials):
        seq_len, nenv, num_edges, h_size = h_spatials.size()
        attn = temporal_embed * spatial_embed
        attn = torch.sum(attn, dim=3)
        temperature = num_edges / np.sqrt(self.attention_size)
        attn = torch.mul(attn, temperature)
        attn = attn.view(seq_len, nenv, self.agent_num, self.human_num)
        attn = torch.nn.functional.softmax(attn, dim=-1)
        h_spatials = h_spatials.view(seq_len, nenv, self.agent_num, self.human_num, h_size)
        h_spatials = h_spatials.view(
            seq_len * nenv * self.agent_num, self.human_num, h_size
        ).permute(0, 2, 1)
        attn = attn.view(seq_len * nenv * self.agent_num, self.human_num).unsqueeze(-1)
        weighted_value = torch.bmm(h_spatials, attn)
        weighted_value = weighted_value.squeeze(-1).view(seq_len, nenv, self.agent_num, h_size)
        return weighted_value, attn

    def forward(self, h_temporal, h_spatials):
        self.human_num = h_spatials.size()[2] // self.agent_num
        weighted_value_list, attn_list = [], []
        for i in range(self.num_attention_head):
            temporal_embed = self.temporal_edge_layer[i](h_temporal)
            spatial_embed = self.spatial_edge_layer[i](h_spatials)
            temporal_embed = temporal_embed.repeat_interleave(self.human_num, dim=2)
            weighted_value, attn = self.att_func(temporal_embed, spatial_embed, h_spatials)
            weighted_value_list.append(weighted_value)
            attn_list.append(attn)
        return weighted_value_list[0], attn_list[0]


class SRNN(nn.Module):
    def __init__(self, obs_space_dict, config, infer=False):
        super().__init__()
        self.infer = infer
        self.is_recurrent = True
        self.config = config
        self.human_num = config.sim.human_num
        self.seq_length = config.ppo.num_steps
        self.nenv = config.training.num_processes
        self.nminibatch = config.ppo.num_mini_batch
        self.human_node_rnn_size = config.SRNN.human_node_rnn_size
        self.human_human_edge_rnn_size = config.SRNN.human_human_edge_rnn_size
        self.output_size = config.SRNN.human_node_output_size

        self.humanNodeRNN = HumanNodeRNN(config)
        self.humanhumanEdgeRNN_spatial = HumanHumanEdgeRNN(config)
        self.humanhumanEdgeRNN_temporal = HumanHumanEdgeRNN(config)
        self.attn = EdgeAttention(config)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        num_inputs = hidden_size = self.output_size
        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
        )
        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
        )
        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        self.robot_linear = init_(nn.Linear(7, 3))
        self.human_node_final_linear = init_(nn.Linear(self.output_size, 2))

        self.num_edges = self.human_num + 1
        self.temporal_edges = [0]
        self.spatial_edges = np.arange(1, self.human_num + 1)

    def forward(self, inputs, rnn_hxs, masks, infer=True):
        # inference path only: seq_length = 1, nenv = self.nenv (set to 1 by caller)
        seq_length = 1
        nenv = self.nenv

        robot_node = reshapeT(inputs["robot_node"], seq_length, nenv)
        temporal_edges = reshapeT(inputs["temporal_edges"], seq_length, nenv)
        spatial_edges = reshapeT(inputs["spatial_edges"], seq_length, nenv)

        hidden_states_node_RNNs = reshapeT(rnn_hxs["human_node_rnn"], 1, nenv)
        hidden_states_edge_RNNs = reshapeT(rnn_hxs["human_human_edge_rnn"], 1, nenv)
        masks = reshapeT(masks, seq_length, nenv)

        all_hidden_states_edge_RNNs = torch.zeros(
            1, nenv, self.num_edges, rnn_hxs["human_human_edge_rnn"].size()[-1]
        )

        hidden_temporal_start_end = hidden_states_edge_RNNs[:, :, self.temporal_edges, :]
        output_temporal, hidden_temporal = self.humanhumanEdgeRNN_temporal(
            temporal_edges, hidden_temporal_start_end, masks
        )
        all_hidden_states_edge_RNNs[:, :, self.temporal_edges, :] = hidden_temporal

        hidden_spatial_start_end = hidden_states_edge_RNNs[:, :, self.spatial_edges, :]
        output_spatial, hidden_spatial = self.humanhumanEdgeRNN_spatial(
            spatial_edges, hidden_spatial_start_end, masks
        )
        all_hidden_states_edge_RNNs[:, :, self.spatial_edges, :] = hidden_spatial

        hidden_attn_weighted, _ = self.attn(output_temporal, output_spatial)

        nodes_current_selected = self.robot_linear(robot_node)
        outputs, h_nodes = self.humanNodeRNN(
            nodes_current_selected, output_temporal, hidden_attn_weighted,
            hidden_states_node_RNNs, masks,
        )

        rnn_hxs["human_node_rnn"] = h_nodes
        rnn_hxs["human_human_edge_rnn"] = all_hidden_states_edge_RNNs

        x = outputs[:, :, 0, :]
        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        for key in rnn_hxs:
            rnn_hxs[key] = rnn_hxs[key].squeeze(0)

        return self.critic_linear(hidden_critic).squeeze(0), hidden_actor.squeeze(0), rnn_hxs


# --- upstream pytorchBaselines/a2c_ppo_acktr/model.py (Policy) --------------
class Policy(nn.Module):
    def __init__(self, obs_shape, action_dim, base_kwargs):
        super().__init__()
        self.base = SRNN(obs_shape, base_kwargs)
        self.srnn = True
        self.dist = DiagGaussian(self.base.output_size, action_dim)

    def act(self, inputs, rnn_hxs, masks, deterministic=True):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks, infer=True)
        dist = self.dist(actor_features)
        action = dist.mode() if deterministic else dist.sample()
        return value, action, rnn_hxs
