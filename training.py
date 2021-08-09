import os, pickle
import tensorflow as tf

def train_pred(m, train, test, dir, n_samples=-1, n_epoch=50, bat_size=32):
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

    train_ds = train.shuffle(10000).take(n_samples).batch(bat_size)
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
    train: training data
    test: test data
    dir: directory to save and load the model
    n_epoch: number of epochs
    bat_size: batch size
    """

    if os.path.exists(dir):
        m = tf.keras.models.load_model(dir)
        return m

    m.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy'])

    h = m.fit(train, train,
              epochs=n_epoch,
              shuffle=True,
              validation_data=(test, test))

    m.save(dir)

    return m

def gen_labels(m, data, bat_size=32):
    """ Generates dataset containing (input, pred) for given model and inputs.

    Parameters
    ----------
    m : model being queried
    data : data used to query model, labeled
    bat_size : batch size
    """

    x_batch = data.map(lambda x,_: x,
                       num_parallel_calls=tf.data.AUTOTUNE).batch(bat_size)
    y_batch = x_batch.map(lambda x: tf.argmax(m(x), axis=1),
                          num_parallel_calls=tf.data.AUTOTUNE)
    x,y = x_batch.unbatch(), y_batch.unbatch()

    return tf.data.Dataset.zip((x,y))


def gen_expl(m, data, expl_fn, expl_spec, dir):
    """ Generates explanation for given model.

    Parameters
    ----------
    m : model being queried
    data : data used to query model
    expl_fn : explanation function
    expl_spec : shape of explanation
    dir : directory to save and load the data
    """

    if not os.path.exists(dir):
        e = data.map(lambda x,y: expl_fn(m,x,y), num_parallel_calls=tf.data.AUTOTUNE)
        tf.data.experimental.save(e, dir)

    e = tf.data.experimental.load(dir, expl_spec)

    return e