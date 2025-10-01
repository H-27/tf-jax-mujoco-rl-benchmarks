import gymnasium as gym
def train_loop(env, library, algo, enable_jit=False):
    pass
def plot_results():
    pass


if __name__ == "__main__":
    # define library and algo
    library = ["tf", "jax"]  # Example to switch between libraries
    algo = ["ppo", "sac", "td3"]  # Example to switch between algorithms

    # choose environment
    env_name = "HalfCheetah-v4" # choose from "HalfCheetah-v4", "Hopper-v4", "Ant-v4", "Walker2d-v4"
    envs = ["HalfCheetah-v4", "Hopper-v4", "Ant-v4", "Walker2d-v4"]  # Example to switch between environments
    env = gym.make(env_name)

    # Jit enabled tf
    jit_for_tf = [True, False]  # Example flag for JIT compilation in TensorFlow
    #tf.config.optimizer.set_jit(True)  # Enable XLA.
    # jit_compile=True argument for model.compile()
    
    # train agent
    for lib in library:
        for algorithm in algo:
            for env_name in envs:
                env = gym.make(env_name)
                train_loop(env, lib, algorithm)
    
    plot_results()  # --- IGNORE ---

