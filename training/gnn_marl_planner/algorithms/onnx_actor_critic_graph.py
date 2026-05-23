import torch
import torch.nn as nn
from algorithms.utils.util import init, check
from algorithms.onnx_act_graph import ACTLayer
from algorithms.utils.gnn_data_onnx import GNNBase
from utils.util import get_shape_from_obs_space



class Actor(nn.Module):
    """
    Actor network class for HAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(Actor, self).__init__()
        self.hidden_size = args.hidden_size
        self.args=args
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(device=device)
        print("args are", args)
        node_obs_shape = get_shape_from_obs_space(obs_space)[0]
    
        base = GNNBase
        
        self.base = base(args, node_obs_shape, edge_dim=0)
        # total_params = sum(p.numel() for p in self.base.parameters())
        # print(f'Total number of parameters in model: {total_params}')

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain, args)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), self._gain)
        # self.act_distances =  nn.Sequential([init_(nn.Linear(self.hidden_size, 1)), nn.Softmax(dim=-1)])
        
        self.to(device)

    def forward(self, x, edge_index, indices_viewpoint, deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """

        
        actor_features, masks = self.base( x, edge_index, indices_viewpoint)
        # print("!!!actor features ", actor_features.shape)
        # print("available_actions", available_actions.shape)
        # actions, action_log_probs, soft_prob = self.act(actor_features, available_actions, deterministic)
        action_logits = self.act(actor_features, masks, deterministic)
        # print("actions shape is that", actions.shape)
        # return actions, action_log_probs, rnn_states, soft_prob
        return action_logits
  