"""Config for the DS-RNN policy in `model/55554.pt`.

Every SRNN size is pinned by that checkpoint's tensor shapes; the remaining
fields match upstream CrowdNav_DSRNN `crowd_nav/configs/config.py`. The ppo and
training entries only shape training-time batches - `planner.py` sets
`policy.base.nenv = 1` for inference.
"""


class BaseConfig:
    pass


class Config:
    SRNN = BaseConfig()
    SRNN.human_node_rnn_size = 128           # humanNodeRNN.gru.weight_ih_l0 (384, 128)
    SRNN.human_human_edge_rnn_size = 256     # humanNodeRNN.edge_embed.weight (64, 256)
    SRNN.human_node_input_size = 3           # humanNodeRNN.encoder_linear.weight (64, 3)
    SRNN.human_human_edge_input_size = 2     # humanhumanEdgeRNN_spatial.encoder_linear.weight (64, 2)
    SRNN.human_node_embedding_size = 64
    SRNN.human_human_edge_embedding_size = 64
    SRNN.human_node_output_size = 256        # humanNodeRNN.output_linear.weight (256, 128)
    SRNN.attention_size = 64                 # attn.temporal_edge_layer.0.weight (64, 256)

    sim = BaseConfig()
    sim.human_num = 5

    ppo = BaseConfig()
    ppo.num_steps = 30
    ppo.num_mini_batch = 2

    training = BaseConfig()
    training.num_processes = 12
