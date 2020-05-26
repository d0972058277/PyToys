from tensorflow.python.client import device_lib
import tensorflow as tf
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

print("Num GPUs Available: ", len(
    tf.config.experimental.list_physical_devices('GPU')))

print(device_lib.list_local_devices())

tf.test.is_gpu_available()
tf.test.is_built_with_cuda()
tf.config.list_physical_devices('GPU')

print(device_lib.list_local_devices())
