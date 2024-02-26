import copy

import ml_collections

from jaxrl_m.typing import *

from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxrl_m.common import TrainState
from jaxrl_m.networks import Policy
from jaxrl_m.vision import encoders

import flax
from flax.core import freeze, unfreeze
from src.special_networks import GoalConditionedValue, GoalConditionedCritic, GoalConditionedPhiValue, HILPNetwork


def expectile_loss(adv, diff, expectile=0.7):
    weight = jnp.where(adv >= 0, expectile, (1 - expectile))
    return weight * (diff**2)


def compute_value_loss(agent, batch, network_params):
    # masks are 0 if terminal, 1 otherwise
    batch['masks'] = 1.0 - batch['rewards']
    # rewards are 0 if terminal, -1 otherwise
    batch['rewards'] = batch['rewards'] - 1.0

    (next_v1, next_v2) = agent.network(batch['next_observations'], batch['goals'], method='target_value')
    next_v = jnp.minimum(next_v1, next_v2)
    q = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v

    (v1_t, v2_t) = agent.network(batch['observations'], batch['goals'], method='target_value')
    v_t = (v1_t + v2_t) / 2
    adv = q - v_t

    q1 = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v1
    q2 = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v2
    (v1, v2) = agent.network(batch['observations'], batch['goals'], method='value', params=network_params)
    v = (v1 + v2) / 2

    value_loss1 = expectile_loss(adv, q1 - v1, agent.config['expectile']).mean()
    value_loss2 = expectile_loss(adv, q2 - v2, agent.config['expectile']).mean()
    value_loss = value_loss1 + value_loss2

    return value_loss, {
        'value_loss': value_loss,
        'v max': v.max(),
        'v min': v.min(),
        'v mean': v.mean(),
        'abs adv mean': jnp.abs(adv).mean(),
        'adv mean': adv.mean(),
        'adv max': adv.max(),
        'adv min': adv.min(),
        'accept prob': (adv >= 0).mean(),
    }


def compute_skill_value_loss(agent, batch, network_params):
    q1, q2 = agent.network(batch['observations'], batch['skills'], batch['actions'], method='skill_target_critic')
    q = jnp.minimum(q1, q2)
    v = agent.network(batch['observations'], batch['skills'], method='skill_value', params=network_params)
    adv = q - v
    value_loss = expectile_loss(adv, q - v, agent.config['skill_expectile']).mean()

    return value_loss, {
        'value_loss': value_loss,
        'v max': v.max(),
        'v min': v.min(),
        'v mean': v.mean(),
        'abs adv mean': jnp.abs(adv).mean(),
        'adv mean': adv.mean(),
        'adv max': adv.max(),
        'adv min': adv.min(),
        'accept prob': (adv >= 0).mean(),
    }


def compute_skill_critic_loss(agent, batch, network_params):
    next_v = agent.network(batch['next_observations'], batch['skills'], method='skill_value')
    q = batch['rewards'] + agent.config['skill_discount'] * next_v  # No 'done'

    q1, q2 = agent.network(batch['observations'], batch['skills'], batch['actions'], method='skill_critic', params=network_params)
    critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()

    return critic_loss, {
        'critic_loss': critic_loss,
        'q max': q.max(),
        'q min': q.min(),
        'q mean': q.mean(),
    }


def compute_skill_actor_loss(agent, batch, network_params):
    v = agent.network(batch['observations'], batch['skills'], method='skill_value')
    q1, q2 = agent.network(batch['observations'], batch['skills'], batch['actions'], method='skill_target_critic')
    q = jnp.minimum(q1, q2)
    adv = q - v

    exp_a = jnp.exp(adv * agent.config['skill_temperature'])
    exp_a = jnp.minimum(exp_a, 100.0)

    dist = agent.network(batch['observations'], batch['skills'], method='skill_actor', params=network_params)
    log_probs = dist.log_prob(batch['actions'])
    actor_loss = -(exp_a * log_probs).mean()

    return actor_loss, {
        'actor_loss': actor_loss,
        'adv': adv.mean(),
        'bc_log_probs': log_probs.mean(),
        'adv_median': jnp.median(adv),
        'mse': jnp.mean((dist.mode() - batch['actions'])**2),
    }


def loss_fn(network_params, agent, batch):
    info = {}

    # GCVF
    value_loss, value_info = compute_value_loss(agent, batch, network_params)
    for k, v in value_info.items():
        info[f'value/{k}'] = v

    # Skill policy
    batch_size = batch['observations'].shape[0]
    batch['phis'] = agent.network(batch['observations'], method='phi')
    batch['next_phis'] = agent.network(batch['next_observations'], method='phi')
    random_skills = np.random.randn(batch_size, agent.config['skill_dim'])
    batch['skills'] = random_skills / jnp.linalg.norm(random_skills, axis=1, keepdims=True)
    batch['rewards'] = ((batch['next_phis'] - batch['phis']) * batch['skills']).sum(axis=1)

    skill_value_loss, skill_value_info = compute_skill_value_loss(agent, batch, network_params)
    for k, v in skill_value_info.items():
        info[f'skill_value/{k}'] = v

    skill_critic_loss, skill_critic_info = compute_skill_critic_loss(agent, batch, network_params)
    for k, v in skill_critic_info.items():
        info[f'skill_critic/{k}'] = v

    skill_actor_loss, skill_actor_info = compute_skill_actor_loss(agent, batch, network_params)
    for k, v in skill_actor_info.items():
        info[f'skill_actor/{k}'] = v

    loss = value_loss + skill_value_loss + skill_critic_loss + skill_actor_loss

    return loss, info


class HILPAgent(flax.struct.PyTreeNode):
    rng: PRNGKey
    network: TrainState
    config: dict = flax.struct.field(pytree_node=False)

    def update(agent, batch):
        new_target_params = jax.tree_map(
            lambda p, tp: p * agent.config['target_update_rate'] + tp * (1 - agent.config['target_update_rate']), agent.network.params['networks_value'], agent.network.params['networks_target_value']
        )
        new_skill_target_params = jax.tree_map(
            lambda p, tp: p * agent.config['target_update_rate'] + tp * (1 - agent.config['target_update_rate']), agent.network.params['networks_skill_critic'], agent.network.params['networks_skill_target_critic']
        )

        new_network, info = agent.network.apply_loss_fn(loss_fn=partial(loss_fn, agent=agent, batch=batch), has_aux=True)

        params = unfreeze(new_network.params)
        params['networks_target_value'] = new_target_params
        params['networks_skill_target_critic'] = new_skill_target_params
        new_network = new_network.replace(params=freeze(params))

        return agent.replace(network=new_network), info
    update = jax.jit(update)

    def get_loss_info(agent, batch):
        loss, info = loss_fn(agent.network.params, agent, batch)

        return info
    get_loss_info = jax.jit(get_loss_info)

    def sample_skill_actions(agent,
                             observations: np.ndarray,
                             skills: np.ndarray = None,
                             *,
                             seed: PRNGKey = None,
                             temperature: float = 1.0) -> jnp.ndarray:
        dist = agent.network(observations, skills, temperature=temperature, method='skill_actor')
        actions = dist.sample(seed=seed)
        actions = jnp.clip(actions, -1, 1)
        return actions
    sample_skill_actions = jax.jit(sample_skill_actions)

    @jax.jit
    def get_phi(agent, s: np.ndarray) -> jnp.ndarray:
        phi = agent.network(s, method='phi')
        return phi


def create_learner(
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        lr: float = 3e-4,
        value_hidden_dims: Sequence[int] = (512, 512, 512),
        actor_hidden_dims: Sequence[int] = (512, 512, 512),
        discount: float = 0.99,
        tau: float = 0.005,
        expectile: float = 0.95,
        use_layer_norm: int = 1,
        skill_dim: int = 32,
        skill_expectile: float = 0.9,
        skill_temperature: float = 10,
        skill_discount: float = 0.99,
        encoder: str = None,
        **kwargs):

        print('Extra kwargs:', kwargs)

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

        if encoder is not None:
            encoder_module = encoders[encoder]
        else:
            encoder_module = None

        value_def = GoalConditionedPhiValue(hidden_dims=value_hidden_dims, use_layer_norm=use_layer_norm, ensemble=True, skill_dim=skill_dim, encoder=encoder_module)

        skill_value_def = GoalConditionedValue(hidden_dims=value_hidden_dims, use_layer_norm=use_layer_norm, ensemble=False, encoder=encoder_module)
        skill_critic_def = GoalConditionedCritic(hidden_dims=value_hidden_dims, use_layer_norm=use_layer_norm, ensemble=True, encoder=encoder_module)
        skill_actor_def = Policy(actor_hidden_dims, action_dim=actions.shape[-1], log_std_min=-5.0, state_dependent_std=False, tanh_squash_distribution=False, encoder=encoder_module)

        network_def = HILPNetwork(
            networks={
                'value': value_def,
                'target_value': copy.deepcopy(value_def),

                'skill_value': skill_value_def,
                'skill_target_value': copy.deepcopy(skill_value_def),
                'skill_critic': skill_critic_def,
                'skill_target_critic': copy.deepcopy(skill_critic_def),
                'skill_actor': skill_actor_def,
            },
        )
        network_tx = optax.adam(learning_rate=lr)
        network_params = network_def.init(value_key, observations, observations, actions, np.zeros((1, skill_dim)))['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)
        params = unfreeze(network.params)
        params['networks_target_value'] = params['networks_value']
        params['networks_skill_target_critic'] = params['networks_skill_critic']
        network = network.replace(params=freeze(params))

        config = flax.core.FrozenDict(dict(
            discount=discount, target_update_rate=tau, expectile=expectile,
            skill_dim=skill_dim, skill_expectile=skill_expectile, skill_temperature=skill_temperature, skill_discount=skill_discount,
        ))

        return HILPAgent(rng, network=network, config=config)
