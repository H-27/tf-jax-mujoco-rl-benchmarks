# tf-jax-mujoco-rl-benchmarks
Performance comparison of JAX-based and TensorFlow-based implementations of common RL algorithms tested across classic MuJoCo tasks.

PPO: strong on-policy baseline, stable and simple.
SAC: off-policy, sample-efficient, robust; great MuJoCo perf.
TD3: off-policy deterministic; faster than SAC in forward cost, good baseline.