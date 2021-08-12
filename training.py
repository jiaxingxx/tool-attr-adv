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
              loss=tf.keras.losses.MeanSquaredError())

    h = m.fit(train, train,
              epochs=n_epoch,
              shuffle=True,
              validation_data=(test, test))

    m.save(dir)

    return m

def gen_labels(m, data, bat_size=128):
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

def gen_attr(m, data, attr_fn, dir):
    """ Generates and returns attributions.

    Parameters
    ----------
    m: model being queried
    data: data used to query model, labeled
    attr_fn: function used to generate attribution
    dir: directory to save and load the attribution data
    """

    if os.path.exists(dir):
        return pickle.load(open(dir,'rb'))

    attr = []
    for x,_ in tqdm(data):
        attr.append(attr_fn(model,x))

    pickle.dump(attr, open(dir,'wb'))
    return attr

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
        return pickle.load(open(dir,'rb'))

    xs, ys = [], []
    for x,_ in data.batch(bat_size):
        x_ae = atk_fn(m, x, **atk_args)
        y_ae = np.argmax(model.predict(x_ae), axis=1)
        xs.append(x_ae)
        ys.append(y_ae)

    ae = (tf.concat(x_ae, axis=0), tf.concat(y_ae, axis=0))

    pickle.dump(ae, open(dir,'wb'))
    return ae