import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

# Optional: avoid TF pre-allocating the whole GPU
try:
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass

tfd = tfp.distributions
tfb = tfp.bijectors


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


class ActorGaussianTF(tf.keras.Model):
    def __init__(self, act_dim, input_dims, alpha, fc1=256, fc2=256, chkpt_dir="tmp/ppo"):
        super().__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, "actor_tf_gaussian_ppo.weights")
        self.backbone = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(*input_dims,)),
                tf.keras.layers.Dense(fc1, activation="relu"),
                tf.keras.layers.Dense(fc2, activation="relu"),
            ]
        )
        self.mu = tf.keras.layers.Dense(act_dim, activation=None)
        # State-independent log_std parameter (learned)
        self.log_std = tf.Variable(tf.zeros((act_dim,), dtype=tf.float32))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)

    def call(self, x, training=False):
        h = self.backbone(x, training=training)
        mu = self.mu(h)
        return mu, self.log_std


class CriticNetworkTF(tf.keras.Model):
    def __init__(self, input_dims, alpha, fc1=256, fc2=256, chkpt_dir="tmp/ppo"):
        super().__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, "critic_tf_gaussian_ppo.weights")
        self.net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(*input_dims,)),
                tf.keras.layers.Dense(fc1, activation="relu"),
                tf.keras.layers.Dense(fc2, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)

    def call(self, x, training=False):
        return self.net(x, training=training)


class AgentTFGaussian:
    def __init__(
        self,
        act_low: np.ndarray,
        act_high: np.ndarray,
        input_dims,
        gamma=0.99,
        alpha=3e-4,
        gae_lambda=0.95,
        policy_clip=0.2,
        batch_size=64,
        n_epochs=10,
        tf_jit: bool = False,
    ):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.tf_jit = tf_jit

        act_low = np.asarray(act_low, dtype=np.float32)
        act_high = np.asarray(act_high, dtype=np.float32)
        self.act_low = tf.constant(act_low)
        self.act_high = tf.constant(act_high)
        self.act_mid = (self.act_low + self.act_high) / 2.0
        self.act_scale = (self.act_high - self.act_low) / 2.0
        self.act_dim = int(act_low.shape[0])

        self.actor = ActorGaussianTF(self.act_dim, input_dims, alpha)
        self.critic = CriticNetworkTF(input_dims, alpha)
        self.memory = PPOMemory(batch_size)

        # Compiled helpers
        self._dist_fn = self._build_dist_fn()
        self._train_step = self._build_train_step()

    def _build_dist_fn(self):
        @tf.function(jit_compile=False)
        def make_dist(mu, log_std):
            std = tf.exp(log_std)
            base = tfd.Independent(tfd.Normal(loc=mu, scale=std), reinterpreted_batch_ndims=1)
            bij = tfb.Chain([tfb.Shift(self.act_mid), tfb.ScaleMatvecDiag(scale_diag=self.act_scale), tfb.Tanh()])
            return tfd.TransformedDistribution(distribution=base, bijector=bij)
        return make_dist

    def _build_train_step(self):
        jit_flag = self.tf_jit

        @tf.function(jit_compile=jit_flag)
        def train_step(states, actions, old_log_probs, adv_b, val_b):
            with tf.GradientTape() as tape_a, tf.GradientTape() as tape_c:
                mu, log_std = self.actor(states)
                dist = self._dist_fn(mu, log_std)
                new_log_probs = dist.log_prob(actions)
                ratio = tf.exp(new_log_probs - old_log_probs)
                weighted = adv_b * ratio
                clipped = tf.clip_by_value(ratio, 1 - self.policy_clip, 1 + self.policy_clip) * adv_b
                actor_loss = -tf.reduce_mean(tf.minimum(weighted, clipped))

                critic_value = tf.squeeze(self.critic(states), axis=-1)
                returns = adv_b + val_b
                critic_loss = tf.reduce_mean(tf.square(returns - critic_value))

            grads_a = tape_a.gradient(actor_loss, self.actor.trainable_variables + [self.actor.log_std])
            self.actor.optimizer.apply_gradients(zip(grads_a, self.actor.trainable_variables + [self.actor.log_std]))

            grads_c = tape_c.gradient(critic_loss, self.critic.trainable_variables)
            self.critic.optimizer.apply_gradients(zip(grads_c, self.critic.trainable_variables))
            return actor_loss, critic_loss

        return train_step

    def remember(self, state, action, log_prob, val, reward, done):
        self.memory.store(state, action, log_prob, val, reward, done)

    def choose_action(self, observation, deterministic: bool = False):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        mu, log_std = self.actor(state)
        dist = self._dist_fn(mu[0], log_std)  # use per-dim log_std
        if deterministic:
            # Deterministic action: mid + tanh(mu) * scale
            a = self.act_mid + tf.tanh(mu[0]) * self.act_scale
            log_prob = dist.log_prob(a)
        else:
            a = dist.sample(seed=None)
            log_prob = dist.log_prob(a)
        value = tf.squeeze(self.critic(state), axis=-1)[0]
        return a.numpy(), float(log_prob.numpy()), float(value.numpy())

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

            adv_tf = tf.convert_to_tensor(advantage, dtype=tf.float32)
            vals_tf = tf.convert_to_tensor(values, dtype=tf.float32)

            for batch in batches:
                states_b = tf.convert_to_tensor(states[batch], dtype=tf.float32)
                actions_b = tf.convert_to_tensor(actions[batch], dtype=tf.float32)
                old_lp_b = tf.convert_to_tensor(old_log_probs[batch], dtype=tf.float32)
                adv_b = tf.gather(adv_tf, batch)
                val_b = tf.gather(vals_tf, batch)
                self._train_step(states_b, actions_b, old_lp_b, adv_b, val_b)

        self.memory.clear()


def _get_spaces_from_env(env):
    """Return (observation_space, action_space) for single or vector envs."""
    obs_space = getattr(env, "single_observation_space", None) or getattr(env, "observation_space", None)
    act_space = getattr(env, "single_action_space", None) or getattr(env, "action_space", None)
    if obs_space is None or act_space is None:
        raise ValueError("Env does not expose observation/action space")
    return obs_space, act_space


def init_tf_gaussian_ppo_from_spaces(observation_space, action_space, **agent_kwargs) -> AgentTFGaussian:
    """Create AgentTFGaussian from Gymnasium spaces.

    agent_kwargs forwards PPO hyperparameters like gamma, alpha, tf_jit, etc.
    """
    obs_dim = int(np.prod(observation_space.shape))
    act_low = np.asarray(action_space.low, dtype=np.float32)
    act_high = np.asarray(action_space.high, dtype=np.float32)
    return AgentTFGaussian(act_low=act_low, act_high=act_high, input_dims=(obs_dim,), **agent_kwargs)


esspace_doc = """Helper doc: init_tf_gaussian_ppo_from_env(env, **kwargs) -> AgentTFGaussian"""

def init_tf_gaussian_ppo_from_env(env, **agent_kwargs) -> AgentTFGaussian:
    """Create AgentTFGaussian from a Gymnasium env (single or vector)."""
    obs_space, act_space = _get_spaces_from_env(env)
    return init_tf_gaussian_ppo_from_spaces(obs_space, act_space, **agent_kwargs)
