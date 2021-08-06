import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'    # silence some tensorflow messages

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import argparse

# make executions deterministic
seed = 0
os.environ['TF_DETERMINISTIC_OPS'] = '0'
tf.random.set_seed(seed)
np.random.seed(seed)

from tqdm import tqdm, trange
import pickle

from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.tf2.attacks.momentum_iterative_method import momentum_iterative_method

# import custom modules
from attribution import *
from models import *
from util import *

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, choices=['cnn'],
                    help="type of the target model")
parser.add_argument("--data", type=str, choices=['mnist'], required=True,
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
parser.add_argument("--gen_attr", action="store_true")
args = parser.parse_args()

# experiment parameters
tar = '_'.join([args.data, args.model])
print(tar)
exit()
data = args.data
eps = args.epsilon
attack = args.attack
attr = args.attr
recons = args.recons
gen_attr = args.gen_attr

# designate gpu
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# enable memory growth
physical_devices = tf.config.list_physical_devices('GPU')
for d in physical_devices:
    tf.config.experimental.set_memory_growth(d, True)

atk_dict = {'fgsm'  : ['fast_gradient_method',
                      {'eps': eps, 'norm': np.inf, 'clip_min':0.0, 'clip_max':1.0}],
            'pgd'   : ['projected_gradient_descent',
                      {'eps': eps, 'eps_iter': 0.01, 'nb_iter': 40, 'norm': np.inf, 'clip_min':0.0, 'clip_max':1.0}],
            'cw'    : ['carlini_wagner_l2',
                      {'clip_min':0.0, 'clip_max':1.0}],
            'mim'   : ['momentum_iterative_method',
                      {'eps': eps, 'clip_min':0.0, 'clip_max':1.0}]}

atk_method, atk_args = atk_dict[attack]

# parameters for training and evaluation
train_batch = 32
latent_dim = 64
order = 1.0

MODEL_DIR = f'experiments/{tar}'
RECONS_DIR = f'experiments/{tar}/recons'
DATA_DIR = f'experiments/{tar}/data'
ADV_DIR = f'experiments/{tar}/data/{attack}_{eps}'
EXP_DIR = f'figures/{tar}_{recons}_{attr}_{attack}_{eps}'

# make directories if path does not exist
dir_names = ['experiments', MODEL_DIR, RECONS_DIR, DATA_DIR, ADV_DIR, 'figures', EXP_DIR]
mkdir(dir_names)

# get dataset
eprint('loading and processing dataset ... ')
dataset = tfds.load(data, data_dir='data', as_supervised=True)
train, test = dataset['train'], dataset['test']

# normalization
train = train.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
test = test.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)

x_train = train.map(get_x, num_parallel_calls=tf.data.AUTOTUNE)
x_test = test.map(get_x, num_parallel_calls=tf.data.AUTOTUNE)

y_train = np.array(list(train.map(get_y, num_parallel_calls=tf.data.AUTOTUNE).as_numpy_iterator()))
y_test = np.array(list(test.map(get_y, num_parallel_calls=tf.data.AUTOTUNE).as_numpy_iterator()))
eprint('done\n')

# target classifier
model = eval(tar)()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

### train target classifier ###
eprint('training target classifier ... ')

if exists(f'{MODEL_DIR}/saved_model.pb'):
    model = tf.keras.models.load_model(MODEL_DIR)

else:
    train_ds = train.shuffle(10000).batch(32)
    test_ds = test.batch(32)

    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    model.fit(train_ds,
              epochs=10,
              shuffle=True,
              validation_data=test_ds)

    model.save(MODEL_DIR)

eprint('done\n')

### generate attributions ###
eprint('generating attributions ... ')

if exists(f'{DATA_DIR}/{attr}_train') and exists(f'{DATA_DIR}/{attr}_test'):
    g_train = pickle.load(open(f'{DATA_DIR}/{attr}_train','rb'))
    g_test = pickle.load(open(f'{DATA_DIR}/{attr}_test','rb'))

else:
    g_train, g_test = [], []

    for x in tqdm(x_train):
        g_train.append(eval(attr)(model,x))

    for x in tqdm(x_test):
        g_test.append(eval(attr)(model,x))

    g_train, g_test = np.array(g_train), np.array(g_test)

    pickle.dump(g_train, open(f'{DATA_DIR}/{attr}_train','wb'))
    pickle.dump(g_test, open(f'{DATA_DIR}/{attr}_test','wb'))

eprint('done\n')

### generate adversarial examples ###
eprint('generating adversarial examples ... ')

if exists(f'{ADV_DIR}/adv_train') and exists(f'{ADV_DIR}/adv_test'):
    (adv_train_x, adv_train_y) = pickle.load(open(f'{ADV_DIR}/adv_train','rb'))
    (adv_test_x, adv_test_y) = pickle.load(open(f'{ADV_DIR}/adv_test','rb'))

else:
    adv_train_x, adv_train_y = [], []
    adv_test_x, adv_test_y = [], []

    for x in x_train.batch(128):
        x_adv = eval(atk_method)(model, x, **atk_args)
        y_adv = np.argmax(model.predict(x_adv), axis=1)
        adv_train_x.append(x_adv)
        adv_train_y.append(y_adv)

    adv_train_x = tf.concat(adv_train_x, axis=0)
    adv_train_y = tf.concat(adv_train_y, axis=0)

    for x in x_test.batch(128):
        x_adv = eval(atk_method)(model, x, **atk_args)
        y_adv = np.argmax(model.predict(x_adv), axis=1)
        adv_test_x.append(x_adv)
        adv_test_y.append(y_adv)

    adv_test_x = tf.concat(adv_test_x, axis=0)
    adv_test_y = tf.concat(adv_test_y, axis=0)

    pickle.dump((adv_train_x, adv_train_y), open(f'{ADV_DIR}/adv_train','wb'))
    pickle.dump((adv_test_x, adv_test_y), open(f'{ADV_DIR}/adv_test','wb'))

eprint('done\n')

### generate explanations for adversarial images ###
eprint('generating attributions for adversarial examples ...\n')

if exists(f'{ADV_DIR}/{attr}_adv_train') and exists(f'{ADV_DIR}/{attr}_adv_test'):
    g_adv_train = pickle.load(open(f'{ADV_DIR}/{attr}_adv_train','rb'))
    g_adv_test = pickle.load(open(f'{ADV_DIR}/{attr}_adv_test','rb'))

else:
    g_adv_train, g_adv_test = [], []

    for x in tqdm(adv_train_x):
        g_adv_train.append(eval(attr)(model,x))

    for x in tqdm(adv_test_x):
        g_adv_test.append(eval(attr)(model,x))

    g_adv_train, g_adv_test = np.array(g_adv_train), np.array(g_adv_test)

    pickle.dump(g_adv_train, open(f'{ADV_DIR}/{attr}_adv_train','wb'))
    pickle.dump(g_adv_test, open(f'{ADV_DIR}/{attr}_adv_test','wb'))

eprint('done\n')

### plotting ###
eprint('plotting ... ')

# plot attributions
plot_results([x_train, g_train, x_test, g_test],
            captions=['x_train',f'{attr}_train','x_test',f'{attr}_test'],
            filename=f'{EXP_DIR}/attr_norm.png')

# plot attributions of adversarial samples
plot_results(imgs=[adv_train_x, g_adv_train, adv_test_x, g_adv_test],
            captions=['adv_train', f'{attr}_adv_train', 'adv_test', f'{attr}_adv_test'],
            filename=f'{EXP_DIR}/attr_adv.png')

eprint('done\n')

exit() if gen_attr else ...

### reconstruction ###
eprint('evaluating ...\n')

N_LABELS = 10

for t_label in range(N_LABELS):

    ### filtering ###
    cond_train = np.logical_and(adv_train_y == t_label, y_train != t_label)
    cond_test = np.logical_and(adv_test_y == t_label, y_test != t_label)

    g_adv_train_t = g_adv_train[cond_train]
    g_adv_test_t = g_adv_test[cond_test]

    g_train_t = g_train[y_train == t_label]
    g_test_t = g_test[y_test == t_label]

    ### train autoencoder for reconstruction ###
    loss_fn = tf.keras.losses.MeanSquaredError()

    autoencoder = eval(recons)(latent_dim)

    if os.path.exists(f'{RECONS_DIR}/{recons}_{attr}_{t_label}'):
        autoencoder = tf.keras.models.load_model(f'{RECONS_DIR}/{recons}_{attr}_{t_label}')

    else:
        autoencoder.compile(optimizer='adam', loss=loss_fn)

        autoencoder.fit(g_train_t, g_train_t,
                        batch_size=train_batch,
                        shuffle=True,
                        epochs=10,
                        validation_data=(g_test_t, g_test_t))

        autoencoder.save(f'{RECONS_DIR}/{recons}_{attr}_{t_label}')

    ### evaluation ###
    eprint(f'... label {t_label} ... ')
    loss_fn = lambda x,y: lp_loss_q(x,y,order=order)

    # predictions
    r_train = autoencoder.predict(g_train_t)
    r_test = autoencoder.predict(g_test_t)
    r_adv_train = autoencoder.predict(g_adv_train_t)
    r_adv_test = autoencoder.predict(g_adv_test_t)

    # losses
    train_loss = loss_fn(flatten_ds(g_train_t), flatten_ds(r_train))
    test_loss = loss_fn(flatten_ds(g_test_t), flatten_ds(r_test))
    train_loss_adv = loss_fn(flatten_ds(g_adv_train_t), flatten_ds(r_adv_train))
    test_loss_adv = loss_fn(flatten_ds(g_adv_test_t), flatten_ds(r_adv_test))

    # threshold
    loss_norm = test_loss
    loss_anom = np.concatenate((train_loss_adv, test_loss_adv))
    thresh = opt_thresh(loss_norm, loss_anom)

    # anomaly detection
    pred_norm_tr = detect(autoencoder, g_train_t, thresh, loss_fn)
    pred_norm_te = detect(autoencoder, g_test_t, thresh, loss_fn)
    pred_anom_tr = detect(autoencoder, g_adv_train_t, thresh, loss_fn)
    pred_anom_te = detect(autoencoder, g_adv_test_t, thresh, loss_fn)

    # prediction and ground truth
    y_score = np.concatenate((test_loss, train_loss_adv, test_loss_adv))
    y_pred = np.concatenate((pred_norm_te, pred_anom_tr, pred_anom_te))
    y_true = np.concatenate((np.zeros_like(pred_norm_te), np.ones_like(pred_anom_tr), np.ones_like(pred_anom_te)))

    # get stats
    exp_conf = [tar, recons, attr, attack, eps, order, t_label]
    exp_stat = get_stats(y_pred, y_true) + (get_score(y_score, y_true), )
    exp_len = [len(g_test_t), len(g_adv_train_t)+len(g_adv_test_t)]

    bench_all = '\t'.join(map(str, exp_conf)) + '\t'
    bench_all += '\t'.join(map(lambda i: f'{i:.4f}', exp_stat)) + '\t'
    bench_all += '\t'.join(map(str, exp_len))
    bench_all += '\n'

    bench_file = './bench_all.tsv'

    with open(bench_file, 'a') as f:
        f.write(bench_all)

    # plotting
    plot_results([g_train_t, r_train, g_test_t, r_test,
                 g_adv_train_t, r_adv_train, g_adv_test_t, r_adv_test],
                captions=[f'{attr}_train', f'recons_train',
                          f'{attr}_test', f'recons_test',
                          f'{attr}_adv_train', f'recons_adv_train',
                          f'{attr}_adv_test', f'recons_adv_test'],
                filename=f'{EXP_DIR}/recons_{t_label}.png')

    eprint('done\n')

eprint('done\n\n')