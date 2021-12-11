import jax

import tensorflow as tf
import tensorflow_datasets as tfds

CROP_PADDING = 32
SEED = 1
tf.random.set_seed(SEED)

def preprocess_for_train(image, image_size, dtype):
  begin, size, _ = tf.image.sample_distorted_bounding_box(
      tf.shape(image),
      tf.zeros([0, 0, 4], tf.float32),
      area_range=(0.05, 1.0),
      min_object_covered=0,
      use_image_if_no_bounding_boxes=True,
      seed=SEED)
  image = tf.slice(image, begin, size)
  image.set_shape([None, None, 3])
  image = tf.image.resize(image, [image_size, image_size])
  image = tf.cast(image, dtype)
  return image

def preprocess_for_eval(image, image_size, dtype):
  shape = tf.shape(image)
  height, width = shape[0], shape[1]
  ratio = image_size / (image_size + CROP_PADDING)
  crop_size = tf.cast(
      (ratio * tf.cast(tf.minimum(height, width), tf.float32)), tf.int32)
  y, x = (height - crop_size) // 2, (width - crop_size) // 2
  image = tf.image.crop_to_bounding_box(image, y, x, crop_size, crop_size)
  image.set_shape([None, None, 3])
  image = tf.image.resize(image, [image_size, image_size])
  image = tf.cast(image, dtype)
  return image

def create_split(dataset_builder, batch_size, image_size, train, dtype, cache=False):
  if train:
    train_examples = dataset_builder.info.splits['train'].num_examples
    split_size = train_examples // jax.process_count()
    start = jax.process_index() * split_size
    split = 'train[{}:{}]'.format(start, start + split_size)
  else:
    validate_examples = dataset_builder.info.splits['validation'].num_examples
    split_size = validate_examples // jax.process_count()
    start = jax.process_index() * split_size
    split = 'validation[{}:{}]'.format(start, start + split_size)

  def preprocess_example(example):
    image = example['image']
    preprocess_fn = preprocess_for_train if train else preprocess_for_eval
    image = preprocess_fn(image, image_size, dtype)
    image = image / 127.5 - 1
    return {'image': image, 'label': example['label']}

  ds = dataset_builder.as_dataset(split=split)
  options = tf.data.Options()
  options.threading.private_threadpool_size = 48
  ds = ds.with_options(options)

  if cache:
    ds = ds.cache()

  if train:
    ds = ds.repeat()
    ds = ds.shuffle(16 * batch_size, seed=SEED)

  ds = ds.map(preprocess_example, num_parallel_calls=-1)
  ds = ds.batch(batch_size, drop_remainder=True)

  if not train:
    ds = ds.repeat()

  ds = ds.prefetch(10)
  return ds
  
