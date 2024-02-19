import tensorflow as tf
import numpy as np

tf.random.set_seed(1)

tf.random.set_seed(1)
t_x = tf.random.uniform([4, 3], dtype=tf.float32)
t_y = tf.range(4)

ds_x = tf.data.Dataset.from_tensor_slices(t_x)
ds_y = tf.data.Dataset.from_tensor_slices(t_y)

ds_joint = tf.data.Dataset.zip((ds_x, ds_y))
# # print(ds_joint)

ds_trans_map_1 = ds_joint.map(lambda x, y: (x + 2.0, y * 3))
ds_trans_map_2 = ds_joint.map(lambda x, y: (tf.square(x), tf.pow(y, 3)))
ds_trans_lambda_1 = ds_joint.map(lambda x, y: (x - 0.5, y * 2))
ds_trans_lambda_2 = ds_joint.map(lambda x, y: (tf.sqrt(x), y + 1))

print("Example 1 using map():")
for example in ds_trans_map_1:
    print('  x: ', example[0].numpy(), '  y: ', example[1].numpy())

print("\nExample 2 using map():")
for example in ds_trans_map_2:
    print('  x: ', example[0].numpy(), '  y: ', example[1].numpy())

print("\nExample 1 using lambda:")
for example in ds_trans_lambda_1:
    print('  x: ', example[0].numpy(), '  y: ', example[1].numpy())

print("\nExample 2 using lambda:")
for example in ds_trans_lambda_2:
    print('  x: ', example[0].numpy(), '  y: ', example[1].numpy())


ds = ds_joint.batch(batch_size=3,
                    drop_remainder=False)
batch_x, batch_y = next(iter(ds))
print('Batch-x: \n', batch_x.numpy())
print('Batch-y:   ', batch_y.numpy())


#[3.]
ds = ds_joint.batch(3).repeat(count=2)
for i,(batch_x, batch_y) in enumerate(ds):
    print(i, batch_x.shape, batch_y.numpy())
#[4.]
ds = ds_joint.repeat(count=2).batch(3)
for i,(batch_x, batch_y) in enumerate(ds):
    print(i, batch_x.shape, batch_y.numpy())
#[5.]
# Order 1: shuffle -> batch -> repeat
tf.random.set_seed(1)
ds = ds_joint.shuffle(4).batch(2).repeat(3)
for i,(batch_x, batch_y) in enumerate(ds):
    print(i, batch_x.shape, batch_y.numpy())
#[6.]
## Order 2: batch -> shuffle -> repeat
tf.random.set_seed(1)
ds = ds_joint.batch(2).shuffle(4).repeat(3)
for i,(batch_x, batch_y) in enumerate(ds):
    print(i, batch_x.shape, batch_y.numpy())
