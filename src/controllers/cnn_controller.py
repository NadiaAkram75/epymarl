from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


class CNNBasicMAC:
    """
    Multi-Agent Controller for CNN pixel observation agents.
    Handles all 3 variants — routes agent_id for Variants B and C.
    """
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self._build_agents(args)
        self.agent_output_type = args.agent_output_type
        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode
        )
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        bs = ep_batch.batch_size
        all_outs = []

        for agent_id in range(self.n_agents):
            agent_obs = ep_batch["obs"][:, t, agent_id]
            h = self.hidden_states[:, agent_id]

            if self.args.agent == "cnn_shared":
                agent_out, new_h = self.agent(agent_obs, h)
            else:
                agent_out, new_h = self.agent(agent_obs, h, agent_id=agent_id)

            all_outs.append(agent_out)
            self.hidden_states[:, agent_id] = new_h

        agent_outs = th.stack(all_outs, dim=1)

        if self.agent_output_type == "pi_logits":
            avail_actions = ep_batch["avail_actions"][:, t]
            if getattr(self.args, "mask_before_softmax", True):
                agent_outs[avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(
            batch_size, self.n_agents, -1
        ).clone()

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(
            th.load("{}/agent.th".format(path),
                    map_location=lambda storage, loc: storage)
        )

    def _build_agents(self, args):
        self.agent = agent_REGISTRY[self.args.agent](0, self.args)