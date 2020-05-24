import tensorflow as tf

dataset = tf.data.Dataset.from_tensors(tf.constant(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], shape=(10, )))
print(dataset)

x_data = tf.data.Dataset.from_tensor_slices(
    tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], shape=(10, )))
print(x_data)

y_data = tf.data.Dataset.from_tensor_slices(tf.constant(
    [0, 2, 4, 6, 8, 10, 12, 14, 16, 18], shape=(10, )))
print(y_data)

for data in dataset:
    print(data)

for data1, data2 in zip(x_data, y_data):
    print('x: {}, y: {}'.format(data1, data2))

for data1, data2 in zip(x_data.take(5), y_data.take(5)):
    print('x: {}, y: {}'.format(data1, data2))

for data1, data2 in zip(x_data.take(12), y_data.take(12)):
    print('x: {}, y: {}'.format(data1, data2))

dataset = tf.data.Dataset.zip((x_data, y_data))
print(dataset)

tf.data.Dataset.range(10).map(lambda x: x*2)

x = tf.data.Dataset.range(10)
y = tf.data.Dataset.range(10).map(lambda x: x*2)

dataset = tf.data.Dataset.zip({"x": x, "y": y})
print(dataset)

for data in dataset.take(10):
    print('x: {}, y: {}'.format(data['x'], data['y']))

dataset = tf.data.Dataset.zip({"x": x, "y": y}).batch(2)

for data in dataset.take(5):
    print('x: {}, y: {}'.format(data['x'], data['y']))

dataset = dataset.shuffle(10)
for data in dataset.take(5):
    print('x: {}, y: {}'.format(data['x'], data['y']))

for data in dataset.take(10):
    print('x: {}, y: {}'.format(data['x'], data['y']))

    
print('-' * 50)
dataset = dataset.repeat(2)
for data in dataset.take(10):
    print('x: {}, y: {}'.format(data['x'], data['y']))