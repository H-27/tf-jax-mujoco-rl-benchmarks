# tf-jax-mujoco-rl-benchmarks
Performance comparison of JAX-based and TensorFlow-based implementations of common RL algorithms tested across classic MuJoCo tasks.

## Project overview (short note)
This project is a small, focused testbed to quantify how much raw samplerate we can get from identical reinforcement learning loops implemented in TensorFlow and JAX on standard MuJoCo control tasks (e.g., HalfCheetah, Hopper, Walker2d, Ant). The goal is not to squeeze out state-of-the-art returns but to make apples-to-apples comparisons of throughput in realistic loops.

Core idea
- Use the same lightweight PPO agent in both stacks: TensorFlow 2 (Keras + TF-Probability bijectors) and JAX/Flax (Optax + Distrax).
- PPO: strong on-policy baseline, stable and simple.
- Keep physics on CPU (Gymnasium + MuJoCo) and run policies on GPU; measure end-to-end steps/sec, not just model FLOPs.
- Provide three micro-benchmarks to isolate bottlenecks:
  1) policy-only (forward pass on synthetic inputs)
  2) env-only (MuJoCo stepping with random actions)
  3) full RL loop (policy choose_action + env.step)
- Favor defaults that “just run”: single script runner, vectorized envs for scale, TF XLA toggle, JAX jitted functions, and headless rendering via EGL when needed.

Design choices for fair comparisons
- Same network shapes and update math on both sides, with tanh-squashed actions mapped to box bounds.
- Gymnasium 0.29 “-v4” tasks for stability; no MJX in the main measurements to avoid mixing physics backends.
- Policy on GPU, env on CPU mirrors common workstation setups; vector envs (e.g., N≈256) stress the pipeline realistically.

What you can run today
- Policy-only: compare TF with/without `tf.function(jit_compile=True)` vs JAX jitted forward.
- Env-only: ceiling for CPU MuJoCo throughput on your machine.
- Full loop: end-to-end steps/sec across control tasks at different vector sizes.

## Outlook: what’s next, and what isn’t
Potential extensions that make sense
- JAX MJX physics as a “ceiling” line: Running MuJoCo-in-JAX (MJX) pushes more of the loop onto GPU and can significantly increase throughput. It’s JAX-only, so it’s best plotted as an upper bound rather than a direct TF vs JAX comparison.
- Keras 3 multi-backend experiment: A single PPO written against Keras Core that can flip between TF and JAX backends. Interesting for developer ergonomics; useful as a secondary comparison, but keep the primary results with minimal hand-written TF/JAX to reduce confounders.
- TF-Agents or higher-level libraries: Adding a TF-Agents PPO/SAC baseline shows framework overhead vs the lightweight implementation. Good for realism, but it mixes in architecture and input pipeline differences.
- More algorithms (SAC/TD3): Off-policy methods stress different parts of the stack (replay buffers, target nets). Great for broader coverage once PPO baselines are solid.
    SAC: off-policy, sample-efficient, robust; great MuJoCo perf.
    TD3: off-policy deterministic; faster than SAC in forward cost, good baseline.
- Logging/plots: Store CSV/JSONL per run (env, backend, vector size, jit flags, steps/s) and ship quick plots to compare across machines/GPUs.

Paths that typically don’t make sense for this comparison
- Mixing physics backends in the main chart (e.g., TF agent with MJX physics): Bridging TF to a JAX-native physics stack introduces data movement and makes the comparison uneven.
- Converting Gym MuJoCo tasks to custom JAX-only versions for the primary results: Great for JAX exploration, but not apples-to-apples with TF.
- Chasing every warning (TF-TRT, plugin factories, NUMA): They’re usually benign here and don’t affect the measured loops.

Practical notes
- Ensure GPU is visible to both frameworks; TF and JAX require compatible CUDA/cuDNN wheels. We pin versions accordingly and enable TF XLA optionally for fairness.
- Headless rendering and some MuJoCo features need EGL/GLVND libraries; the dev container sets this up and uses `MUJOCO_GL=egl`.

In short, this repo provides a clean PPO baseline and three complementary micro-benchmarks to make TF vs JAX performance differences easy to measure and easy to reason about. Add MJX, Keras 3, TF-Agents, and off-policy algorithms later as clearly labeled variants rather than mixing them into the core comparison.