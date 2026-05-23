import torch
import torch.nn as nn
from algorithms.utils.util import init, check
from algorithms.utils.act_graph import ACTLayer
from algorithms.utils.gnn_data import GNNBase
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

    def forward(self, actor_data, indices_viewpoint, deterministic=False):
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
        actor_data_list= []
        for episode_actor_data in actor_data:
            data = episode_actor_data
            actor_data_list.append(data)
        
        actor_features, masks = self.base(actor_data_list, indices_viewpoint)
        # print("!!!actor features ", actor_features.shape)
        # print("available_actions", available_actions.shape)
        # actions, action_log_probs, soft_prob = self.act(actor_features, available_actions, deterministic)
        actions, action_log_probs = self.act(actor_features, masks, deterministic)
        # print("actions shape is that", actions.shape)
        # return actions, action_log_probs, rnn_states, soft_prob
        return actions, action_log_probs
    def evaluate_actions(self, actor_data, indices_viewpoint, action, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        # print("avaliable actions", available_actions.shape)
        # actor_data = actor_data.to(**self.tpdv)
        actor_data_list= []
        for episode_actor_data in actor_data:
            data = episode_actor_data.to(**self.tpdv)
            actor_data_list.append(data)

        indices_viewpoint = indices_viewpoint # 这个会在网络中处理,把内部的东西变成tensor, 所以在此处不用上传
        action = check(action).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features, masks = self.base(actor_data_list, indices_viewpoint)

        if self.args.algorithm_name=="hatrpo":
            action_log_probs, dist_entropy ,action_mu, action_std, all_probs= self.act.evaluate_actions_trpo(actor_features,
                                                                    action, available_actions,
                                                                    active_masks=
                                                                    active_masks if self._use_policy_active_masks
                                                                    else None)

            return action_log_probs, dist_entropy, action_mu, action_std, all_probs
        else:
            action_log_probs, dist_entropy = self.act.evaluate_actions(x=actor_features,
                                                                    action=action, available_actions=masks)

            return action_log_probs, dist_entropy

