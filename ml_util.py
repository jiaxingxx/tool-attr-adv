import os, argparse
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

def exp_from_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str,
                        help="type of the target model")
    parser.add_argument("--data", type=str, required=True,
                        help="evaluation dataset")
    parser.add_argument("--attack", type=str, required=True,
                        help="attack method used to generate adversarial examples")
    parser.add_argument("--epsilon", type=float, required=True,
                        help="value of epsilon for adversarial attacks")
    parser.add_argument("--attr", type=str, required=True,
                        help="type of attribution method")
    parser.add_argument("--recons", type=str, required=True,
                        help="type of reconstruction method")
    parser.add_argument("--gpu", type=str, required=True,
                        help="gpu to use")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed to use")

    ret = parser.parse_args()

    set_gpu(ret.gpu)
    set_seed(ret.seed)

    # names for target and reconstruction models
    ret.target = '_'.join([ret.data, ret.model])

    # target, reconstruction, test datasets
    ret.train, ret.test = load_dataset(ret.data)
    ret.input_shape = ret.train.element_spec[0].shape
    ret.attr_shape = (*ret.input_shape[:-1],1)
    ret.attr_spec = tf.TensorSpec(shape=ret.attr_shape, dtype=float)

    return ret


def load_dataset(data_name):
    d_tr = tfds.load(data_name,
                     data_dir='data',
                     split='train',
                     as_supervised=True)
    d_te = tfds.load(data_name,
                     data_dir='data',
                     split='test',
                     as_supervised=True)

    d_tr = d_tr.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    d_te = d_te.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)

    return (d_tr, d_te)


def set_seed(seed):
    # make executions deterministic
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(seed)
    np.random.seed(seed)


def set_gpu(gpu):
    # configure gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    for dev in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(dev, True)

@tf.function
def normalize(img, label):
    return tf.cast(img, tf.float32) / 255., label

@tf.function
def normalize_i(img, label):
    return tf.cast(img, tf.float32) / 255.