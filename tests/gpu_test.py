# import tensorflow
import jax
import tensorflow as tf

if __name__ == "__main__":
    # test tf gpu availability
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPUs available: {gpus}")
    else:
        print("No GPUs available")

    # test jax gpu availability
    jax_devices = jax.devices()
    if jax_devices:
        print(f"JAX devices available: {jax_devices}")
    else:
        print("No JAX devices available")
