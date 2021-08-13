import os
import tensorflow as tf
import numpy as np

from tqdm import tqdm

def train_pred(m, train, test, dir, n_epoch=50, bat_size=32):
    """ Trains a given model from predictions.

    Parameters
    ----------

    m: model being trained
    train: training data, labeled
    test: test data, labeled
    dir: directory to save and load the model
    n_epoch: number of epochs
    bat_size: batch size
    """

    if os.path.exists(dir):
        m = tf.keras.models.load_model(dir)
        return m

    train_ds = train.shuffle(10000).batch(bat_size)
    test_ds = test.batch(bat_size)

    m.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    h = m.fit(train_ds,
              epochs=n_epoch,
              shuffle=True,
              validation_data=test_ds)

    m.save(dir)

    return m


def train_recons(m, train, test, dir, n_samples=-1, n_epoch=30, bat_size=32):
    """ Trains an autoencoder which reconstructs given input.

    Parameters
    ----------

    m: model being trained
    train: training dataset
    test: test dataset
    dir: directory to save and load the model
    n_epoch: number of epochs
    bat_size: batch size
    """

    if os.path.exists(dir):
        m = tf.keras.models.load_model(dir)
        return m

    #train_ds = tf.data.Dataset.zip((train,train)).shuffle(10000).batch(bat_size)
    #test_ds = tf.data.Dataset.zip((test,test)).batch(bat_size)

    m.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.MeanSquaredError())

    h = m.fit(train,train,
              epochs=n_epoch,
              shuffle=True,
              validation_data=(test,test))

    m.save(dir)

    return m


def gen_pred(m, data, bat_size=128):
    """ Generates dataset containing pred for given model and inputs.

    Parameters
    ----------
    m : model being queried
    data : data used to query model, labeled
    bat_size : batch size
    """

    ds = data.batch(bat_size)
    pred = ds.map(lambda x,_: tf.argmax(m(x), axis=1),
                  num_parallel_calls=tf.data.AUTOTUNE)

    return pred.unbatch()


def gen_attr(m, data, attr_fn, attr_spec, dir):
    """ Generates and returns attributions.

    Parameters
    ----------
    m: model being queried
    data: data used to query model, labeled
    attr_fn: function used to generate attribution
    attr_spec: tensorspec for attribution
    dir: directory to save and load the attribution data
    """

    if os.path.exists(dir):
        return tf.data.experimental.load(dir, element_spec=attr_spec)

    attr = []
    for x,_ in tqdm(data):
        attr.append(attr_fn(m,x))

    ds = tf.data.Dataset.from_tensor_slices(attr)
    tf.data.experimental.save(ds, dir)

    return ds


def gen_ae(m, data, atk_fn, atk_args, dir, bat_size=128):
    """ Generates and returns adversarial examples.

    Parameters
    ----------
    m: model being queried
    data: data used to query model, labeled
    atk_fn: adversarial attack function
    atk_args: arguments for adversarial attack
    dir: directory to save and load the attribution data
    """

    if os.path.exists(dir):
        return tf.data.experimental.load(dir, element_spec=data.element_spec)

    xs, ys = [], []
    for x,_ in tqdm(data.batch(bat_size)):
        x_ae = atk_fn(m, x, **atk_args)
        y_ae = np.argmax(m.predict(x_ae), axis=1)
        xs.append(x_ae)
        ys.append(y_ae)

    ds_x = tf.data.Dataset.from_tensor_slices(tf.concat(xs, axis=0))
    ds_y = tf.data.Dataset.from_tensor_slices(tf.concat(ys, axis=0))

    ds = tf.data.Dataset.zip((ds_x,ds_y))
    tf.data.experimental.save(ds, dir)

    return ds