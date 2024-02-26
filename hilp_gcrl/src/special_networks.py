import functools

from jaxrl_m.dataset import Dataset
from jaxrl_m.typing import *
from jaxrl_m.networks import *
import jax


class LayerNormMLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    activate_final: int = False
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_init()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:                
                x = self.activations(x)
                x = nn.LayerNorm()(x)
        return x


class LayerNormRepresentation(nn.Module):
    hidden_dims: tuple = (256, 256)
    activate_final: bool = True
    ensemble: bool = True

    @nn.compact
    def __call__(self, observations):
        module = LayerNormMLP
        if self.ensemble:
            module = ensemblize(module, 2)
        return module(self.hidden_dims, activate_final=self.activate_final)(observations)


class Representation(nn.Module):
    hidden_dims: tuple = (256, 256)
    activate_final: bool = True
    ensemble: bool = True

    @nn.compact
    def __call__(self, observations):
        module = MLP
        if self.ensemble:
            module = ensemblize(module, 2)
        return module(self.hidden_dims, activate_final=self.activate_final, activations=nn.gelu)(observations)


class GoalConditionedValue(nn.Module):
    hidden_dims: tuple = (256, 256)
    readout_size: tuple = (256,)
    use_layer_norm: bool = True
    ensemble: bool = True
    encoder: nn.Module = None

    def setup(self) -> None:
        repr_class = LayerNormRepresentation if self.use_layer_norm else Representation
        value_net = repr_class((*self.hidden_dims, 1), activate_final=False, ensemble=self.ensemble)
        if self.encoder is not None:
            value_net = nn.Sequential([self.encoder(), value_net])
        self.value_net = value_net

    def __call__(self, observations, goals=None, info=False):
        if goals is None:
            v = self.value_net(observations).squeeze(-1)
        else:
            v = self.value_net(jnp.concatenate([observations, goals], axis=-1)).squeeze(-1)

        return v


class GoalConditionedPhiValue(nn.Module):
    hidden_dims: tuple = (256, 256)
    readout_size: tuple = (256,)
    skill_dim: int = 2
    use_layer_norm: bool = True
    ensemble: bool = True
    encoder: nn.Module = None

    def setup(self) -> None:
        repr_class = LayerNormRepresentation if self.use_layer_norm else Representation
        phi = repr_class((*self.hidden_dims, self.skill_dim), activate_final=False, ensemble=self.ensemble)
        if self.encoder is not None:
            phi = nn.Sequential([self.encoder(), phi])
        self.phi = phi

    def get_phi(self, observations):
        return self.phi(observations)[0]  # Use the first vf

    def __call__(self, observations, goals=None, info=False):
        phi_s = self.phi(observations)
        phi_g = self.phi(goals)
        squared_dist = ((phi_s - phi_g) ** 2).sum(axis=-1)
        v = -jnp.sqrt(jnp.maximum(squared_dist, 1e-6))

        return v


class GoalConditionedCritic(nn.Module):
    hidden_dims: tuple = (256, 256)
    readout_size: tuple = (256,)
    use_layer_norm: bool = True
    ensemble: bool = True
    encoder: nn.Module = None

    def setup(self) -> None:
        repr_class = LayerNormRepresentation if self.use_layer_norm else Representation
        critic_net = repr_class((*self.hidden_dims, 1), activate_final=False, ensemble=self.ensemble)
        if self.encoder is not None:
            critic_net = nn.Sequential([self.encoder(), critic_net])
        self.critic_net = critic_net

    def __call__(self, observations, goals=None, actions=None, info=False):
        if goals is None:
            q = self.critic_net(jnp.concatenate([observations, actions], axis=-1)).squeeze(-1)
        else:
            q = self.critic_net(jnp.concatenate([observations, goals, actions], axis=-1)).squeeze(-1)

        return q


def get_rep(
        encoder: nn.Module, targets: jnp.ndarray, bases: jnp.ndarray = None,
):
    if encoder is None:
        return targets
    else:
        if bases is None:
            return encoder(targets)
        else:
            return encoder(targets, bases)


class HILPNetwork(nn.Module):
    networks: Dict[str, nn.Module]

    def unsqueeze_context(self, observations, contexts):
        if len(observations.shape) <= 2:
            return contexts
        else:
            # observations: (H, W, D) or (B, H, W, D)
            # contexts: (Z) -> (H, W, Z) or (B, Z) -> (B, H, W, Z)
            assert len(observations.shape) == len(contexts.shape) + 2
            return jnp.expand_dims(jnp.expand_dims(contexts, axis=-2), axis=-2).repeat(observations.shape[-3], axis=-3).repeat(observations.shape[-2], axis=-2)

    def value(self, observations, goals=None, **kwargs):
        return self.networks['value'](observations, goals, **kwargs)

    def target_value(self, observations, goals=None, **kwargs):
        return self.networks['target_value'](observations, goals, **kwargs)

    def phi(self, observations, **kwargs):
        return self.networks['value'].get_phi(observations, **kwargs)

    def skill_value(self, observations, skills, **kwargs):
        skills = self.unsqueeze_context(observations, skills)
        return self.networks['skill_value'](observations, skills, **kwargs)

    def skill_target_value(self, observations, skills, **kwargs):
        skills = self.unsqueeze_context(observations, skills)
        return self.networks['skill_target_value'](observations, skills, **kwargs)

    def skill_critic(self, observations, skills, actions=None, **kwargs):
        skills = self.unsqueeze_context(observations, skills)
        actions = self.unsqueeze_context(observations, actions)
        return self.networks['skill_critic'](observations, skills, actions, **kwargs)

    def skill_target_critic(self, observations, skills, actions=None, **kwargs):
        skills = self.unsqueeze_context(observations, skills)
        actions = self.unsqueeze_context(observations, actions)
        return self.networks['skill_target_critic'](observations, skills, actions, **kwargs)

    def skill_actor(self, observations, skills, **kwargs):
        skills = self.unsqueeze_context(observations, skills)
        return self.networks['skill_actor'](jnp.concatenate([observations, skills], axis=-1), **kwargs)

    def __call__(self, observations, goals, actions, skills):
        # Only for initialization
        rets = {
            'value': self.value(observations, goals),
            'target_value': self.target_value(observations, goals),
            'skill_actor': self.skill_actor(observations, skills),
            'skill_value': self.skill_value(observations, skills),
            'skill_critic': self.skill_critic(observations, skills, actions),
            'skill_target_critic': self.skill_target_critic(observations, skills, actions),
        }
        return rets
