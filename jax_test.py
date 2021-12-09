import jax
import jax.numpy as jnp

from models import ResNet50

rng = jax.random.PRNGKey(0)

m = ResNet50(num_classes=10)

dummy_input = jnp.ones([8, 224, 224, 3])

variables = m.init(rng, dummy_input)

#print(variables.keys())
print(variables['batch_stats'])
