import os
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import distrax
from flax import struct, serialization
from dataclasses import dataclass
from typing import Tuple


class PPOMemory:
    def __init__(self, batch_size: int):
        self.states, self.actions, self.log_probs = [], [], []
        self.vals, self.rewards, self.dones = [], [], []
        self.batch_size = batch_size

    def store(self, state, action, log_prob, val, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)

    def generate_batches(self):
        n = len(self.states)
        batch_start = np.arange(0, n, self.batch_size)
        indices = np.arange(n, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i : i + self.batch_size] for i in batch_start]
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.log_probs),
            np.array(self.vals),
            np.array(self.rewards),
            np.array(self.dones),
            batches,
        )

    def clear(self):
        self.states, self.actions, self.log_probs = [], [], []
        self.vals, self.rewards, self.dones = [], [], []


class ActorGaussianFlax(nn.Module):
    act_dim: int
    obs_dim: int
    fc1: int = 256
    fc2: int = 256

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        h = nn.relu(nn.Dense(self.fc1)(x))
        h = nn.relu(nn.Dense(self.fc2)(h))
        mu = nn.Dense(self.act_dim)(h)
        # log_std is a free parameter (state-independent)
        log_std = self.param("log_std", nn.initializers.zeros, (self.act_dim,))
        return mu, log_std


class CriticNetworkFlax(nn.Module):
    obs_dim: int
    fc1: int = 256
    fc2: int = 256

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = nn.relu(nn.Dense(self.fc1)(x))
        h = nn.relu(nn.Dense(self.fc2)(h))
        v = nn.Dense(1)(h)
        return v  # [B,1]


@dataclass
class AgentFlaxGaussian:
    act_low: np.ndarray
    act_high: np.ndarray
    input_dims: Tuple[int, ...]
    gamma: float = 0.99
    alpha: float = 3e-4
    gae_lambda: float = 0.95
    policy_clip: float = 0.2
    batch_size: int = 64
    n_epochs: int = 10

    # runtime
    actor_apply: callable = None
    critic_apply: callable = None
    actor_params: dict = None
    critic_params: dict = None
    actor_opt: optax.GradientTransformation = None
    critic_opt: optax.GradientTransformation = None
    actor_opt_state: optax.OptState = None
    critic_opt_state: optax.OptState = None
    memory: PPOMemory = None
    key: jax.Array = None

    def __post_init__(self):
        obs_dim = int(np.prod(self.input_dims))
        self.obs_dim = obs_dim
        self.act_low = jnp.asarray(self.act_low, dtype=jnp.float32)
        self.act_high = jnp.asarray(self.act_high, dtype=jnp.float32)
        self.mid = (self.act_low + self.act_high) / 2.0
        self.scale = (self.act_high - self.act_low) / 2.0
        act_dim = int(self.act_low.shape[0])
        self.act_dim = act_dim

        key = jax.random.PRNGKey(0)
        self.key = key

        actor = ActorGaussianFlax(act_dim, obs_dim)
        critic = CriticNetworkFlax(obs_dim)

        dummy = jnp.zeros((1, obs_dim), dtype=jnp.float32)
        actor_params = actor.init(key, dummy)
        critic_params = critic.init(key, dummy)

        self.actor_apply = actor.apply
        self.critic_apply = critic.apply
        self.actor_params = actor_params
        self.critic_params = critic_params

        self.actor_opt = optax.adam(self.alpha)
        self.critic_opt = optax.adam(self.alpha)
        self.actor_opt_state = self.actor_opt.init(self.actor_params)
        self.critic_opt_state = self.critic_opt.init(self.critic_params)

        self.memory = PPOMemory(self.batch_size)

        # JIT loss/grad
        def loss_fn(actor_params, critic_params, states, actions, old_log_probs, adv, vals):
            mu, log_std = self.actor_apply(actor_params, states)
            std = jnp.exp(log_std)
            base = distrax.MultivariateNormalDiag(loc=mu, scale_diag=std)
            # Tanh squash to bounds: y = mid + tanh(x) * scale
            dist = distrax.Transformed(base, distrax.Chain([distrax.Block(distrax.ScalarAffine(self.scale, self.mid), ndims=1), distrax.Tanh()]))

            new_log_probs = dist.log_prob(actions)
            ratio = jnp.exp(new_log_probs - old_log_probs)
            weighted = adv * ratio
            clipped = jnp.clip(ratio, 1.0 - self.policy_clip, 1.0 + self.policy_clip) * adv
            actor_loss = -jnp.mean(jnp.minimum(weighted, clipped))

            values_pred = jnp.squeeze(self.critic_apply(critic_params, states), axis=-1)
            returns = adv + vals
            critic_loss = jnp.mean((returns - values_pred) ** 2)
            total = actor_loss + 0.5 * critic_loss
            return total, (actor_loss, critic_loss)

        self._grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True))

    def remember(self, state, action, log_prob, val, reward, done):
        self.memory.store(state, action, log_prob, val, reward, done)

    def choose_action(self, observation, deterministic: bool = False):
        x = jnp.asarray([observation], dtype=jnp.float32).reshape(1, self.obs_dim)
        mu, log_std = self.actor_apply(self.actor_params, x)
        std = jnp.exp(log_std)
        base = distrax.MultivariateNormalDiag(loc=mu[0], scale_diag=std)
        dist = distrax.Transformed(base, distrax.Chain([distrax.Block(distrax.ScalarAffine(self.scale, self.mid), ndims=1), distrax.Tanh()]))
        if deterministic:
            a = self.mid + jnp.tanh(mu[0]) * self.scale
            lp = dist.log_prob(a)
        else:
            self.key, sub = jax.random.split(self.key)
            a = dist.sample(seed=sub)
            lp = dist.log_prob(a)
        v = jnp.squeeze(self.critic_apply(self.critic_params, x), axis=-1)[0]
        return np.asarray(a), float(lp), float(v)

    def learn(self):
        for _ in range(self.n_epochs):
            states, actions, old_log_probs, vals_arr, rewards, dones, batches = self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(rewards), dtype=np.float32)
            for t in range(len(rewards) - 1):
                discount, a_t = 1.0, 0.0
                for k in range(t, len(rewards) - 1):
                    a_t += discount * (rewards[k] + self.gamma * values[k + 1] * (1 - int(dones[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t

            for batch in batches:
                states_b = jnp.asarray(states[batch], dtype=jnp.float32).reshape((-1, self.obs_dim))
                actions_b = jnp.asarray(actions[batch], dtype=jnp.float32)
                old_lp_b = jnp.asarray(old_log_probs[batch], dtype=jnp.float32)
                adv_b = jnp.asarray(advantage[batch], dtype=jnp.float32)
                val_b = jnp.asarray(values[batch], dtype=jnp.float32)

                (losses, (g_actor, g_critic)) = self._grad_fn(
                    self.actor_params, self.critic_params, states_b, actions_b, old_lp_b, adv_b, val_b
                )
                (total_loss, (actor_loss, critic_loss)) = losses
                updates_a, self.actor_opt_state = self.actor_opt.update(g_actor, self.actor_opt_state)
                self.actor_params = optax.apply_updates(self.actor_params, updates_a)
                updates_c, self.critic_opt_state = self.critic_opt.update(g_critic, self.critic_opt_state)
                self.critic_params = optax.apply_updates(self.critic_params, updates_c)

        self.memory.clear()


def _get_spaces_from_env(env):
    """Return (observation_space, action_space) for single or vector envs."""
    obs_space = getattr(env, "single_observation_space", None) or getattr(env, "observation_space", None)
    act_space = getattr(env, "single_action_space", None) or getattr(env, "action_space", None)
    if obs_space is None or act_space is None:
        raise ValueError("Env does not expose observation/action space")
    return obs_space, act_space


def init_flax_gaussian_ppo_from_spaces(observation_space, action_space, **agent_kwargs) -> AgentFlaxGaussian:
    """Create AgentFlaxGaussian from Gymnasium spaces.

    agent_kwargs forwards PPO hyperparameters like gamma, alpha, etc.
    """
    obs_dim = int(np.prod(observation_space.shape))
    act_low = np.asarray(action_space.low, dtype=np.float32)
    act_high = np.asarray(action_space.high, dtype=np.float32)
    return AgentFlaxGaussian(act_low=act_low, act_high=act_high, input_dims=(obs_dim,), **agent_kwargs)


def init_flax_gaussian_ppo_from_env(env, **agent_kwargs) -> AgentFlaxGaussian:
    """Create AgentFlaxGaussian from a Gymnasium env (single or vector)."""
    obs_space, act_space = _get_spaces_from_env(env)
    return init_flax_gaussian_ppo_from_spaces(obs_space, act_space, **agent_kwargs)
