from absl.testing import absltest

import jax
import jax.numpy as jnp

import models

jax.config.update('jax_disable_most_optimizations', True)

class ResNet50Test(absltest.TestCase):

  def test_resnet_model(self):
    rng = jax.random.PRNGKey(0)
    model_def = models.ResNet50(num_classes=1000, dtype=jnp.float32)
    variables = model_def.init(rng, jnp.ones([8, 224, 224, 3]))

    self.assertLen(variables, 2)
    # Initial conv and batch_norm -> 2
    # BottleneckResNetBlock in stages: [3, 4, 6, 3] -> 16
    # Followed by a Dense layer -> 1
    self.assertLen(variables['params'], 19)

if __name__ == '__main__':
  absltest.main()
