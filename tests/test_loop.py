import time
import numpy as np
import gymnasium as gym
    

def make_agent(env, backend: str):
    if backend == "tf":
        from agents.tf_ppo_gaussian import init_tf_gaussian_ppo_from_env
        # Enable/disable XLA for comparison via tf_jit flag
        agent = init_tf_gaussian_ppo_from_env(env, alpha=3e-4, batch_size=64, n_epochs=1, tf_jit=False)
    elif backend == "jax":
        from agents.jax_ppo_gaussian import init_flax_gaussian_ppo_from_env
        agent = init_flax_gaussian_ppo_from_env(env, alpha=3e-4, batch_size=64, n_epochs=1)
    else:
        raise ValueError("backend must be 'tf' or 'jax'")
    return agent


def run_loop(env_id: str, backend: str, total_steps: int, seed: int = 0):
    env = gym.make(env_id)
    obs, _ = env.reset(seed=seed)
    agent = make_agent(env, backend)

    ep_returns = []
    ep_ret = 0.0
    t0 = time.perf_counter()

    for t in range(total_steps):
        action, logp, value = agent.choose_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        # Store minimal fields for completeness (training optional)
        agent.remember(obs, action, logp, value, reward, done)
        ep_ret += float(reward)
        if done:
            ep_returns.append(ep_ret)
            ep_ret = 0.0
            obs, _ = env.reset()

    t1 = time.perf_counter()
    env.close()

    sps = total_steps / max(1e-9, (t1 - t0))
    avg_ret = (np.mean(ep_returns) if ep_returns else ep_ret)
    print(f"[test-loop] backend={backend} env={env_id} steps={total_steps} -> {sps:,.0f} steps/s, avg_ep_return={avg_ret:.1f}")


if __name__ == "__main__":
    # define library and algo
    library = "tf" # choose between "tf" and "jax"
    algo = "ppo" #  choose from "ppo", "dqn", "a2c"

    # choose environment
    env_name = "HalfCheetah-v4" # choose from "HalfCheetah-v4", "Hopper-v4", "Ant-v4", "Walker2d-v4"
    steps = 2000
    seed = 0
    run_loop(env_name, library, steps, seed)
    # train agent



