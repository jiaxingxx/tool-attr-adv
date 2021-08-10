from util import *
from ml_util import *

from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.basic_iterative_method import basic_iterative_method
from cleverhans.tf2.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.tf2.attacks.momentum_iterative_method import momentum_iterative_method

# import custom modules
from attribution import *
from models import *
from training import *

# experiment arguments
e = exp_from_arguments()

# attack parameters
atk_dict = {'fgsm'  : ['fast_gradient_method',
                      {'eps': e.epsilon, 'norm': np.inf, 'clip_min':0.0, 'clip_max':1.0}],
            'pgd'   : ['projected_gradient_descent',
                      {'eps': e.epsilon, 'eps_iter': 0.01, 'nb_iter': 40, 'norm': np.inf, 'clip_min':0.0, 'clip_max':1.0}],
            'bim'   : ['basic_iterative_method',
                      {'eps': e.epsilon, 'eps_iter': 0.01, 'nb_iter': 40, 'norm': np.inf, 'clip_min':0.0, 'clip_max':1.0, 'sanity_checks': False}],
            'cw'    : ['carlini_wagner_l2',
                      {'clip_min':0.0, 'clip_max':1.0}],
            'mim'   : ['momentum_iterative_method',
                      {'eps': e.epsilon, 'clip_min':0.0, 'clip_max':1.0}]}

atk_method, atk_args = atk_dict[e.attack]

# parameters for training and evaluation
bat_size = 32
latent_dim = 64
order = 1.0

MODEL_DIR = f'experiments/{e.target}'
RECONS_DIR = f'experiments/{e.target}/recons'
DATA_DIR = f'experiments/{e.target}/data'
ADV_DIR = f'experiments/{e.target}/data/{e.attack}_{e.epsilon}'
EXP_DIR = f'figures/{e.target}_{e.recons}_{e.attr}_{e.attack}_{e.epsilon}'

# make directories if path does not exist
dir_names = ['experiments', MODEL_DIR, RECONS_DIR, DATA_DIR, ADV_DIR, 'figures', EXP_DIR]
mkdir(dir_names)

# processing dataset
eprint('processing dataset ... ')
x_train = e.train.map(get_x, num_parallel_calls=tf.data.AUTOTUNE)
x_test = e.test.map(get_x, num_parallel_calls=tf.data.AUTOTUNE)

y_train = np.array(list(e.train.map(get_y, num_parallel_calls=tf.data.AUTOTUNE).as_numpy_iterator()))
y_test = np.array(list(e.test.map(get_y, num_parallel_calls=tf.data.AUTOTUNE).as_numpy_iterator()))

eprint('done\n')

### train target classifier ###
eprint('configuring target classifier ... ')

model = train_pred(eval(e.target)(),
                   train=e.train,
                   test=e.test,
                   dir=f'{MODEL_DIR}/target',
                   bat_size=bat_size)

eprint('done\n')

### generate attributions ###
eprint('generating attributions ... ')

if exists(f'{DATA_DIR}/{e.attr}_train') and exists(f'{DATA_DIR}/{e.attr}_test'):
    g_train = pickle.load(open(f'{DATA_DIR}/{e.attr}_train','rb'))
    g_test = pickle.load(open(f'{DATA_DIR}/{e.attr}_test','rb'))
else:
    g_train, g_test = [], []

    for x,_ in tqdm(e.train):
        g_train.append(eval(e.attr)(model,x))

    for x,_ in tqdm(e.test):
        g_test.append(eval(e.attr)(model,x))

    g_train, g_test = np.array(g_train), np.array(g_test)

    pickle.dump(g_train, open(f'{DATA_DIR}/{e.attr}_train','wb'))
    pickle.dump(g_test, open(f'{DATA_DIR}/{e.attr}_test','wb'))

eprint('done\n')

### generate adversarial examples ###
eprint('generating adversarial examples ... ')

if exists(f'{ADV_DIR}/adv_train') and exists(f'{ADV_DIR}/adv_test'):
    (adv_train_x, adv_train_y) = pickle.load(open(f'{ADV_DIR}/adv_train','rb'))
    (adv_test_x, adv_test_y) = pickle.load(open(f'{ADV_DIR}/adv_test','rb'))

else:
    adv_train_x, adv_train_y = [], []
    adv_test_x, adv_test_y = [], []

    for x,_ in e.train.batch(128):
        x_adv = eval(atk_method)(model, x, **atk_args)
        y_adv = np.argmax(model.predict(x_adv), axis=1)
        adv_train_x.append(x_adv)
        adv_train_y.append(y_adv)

    adv_train_x = tf.concat(adv_train_x, axis=0)
    adv_train_y = tf.concat(adv_train_y, axis=0)

    for x,_ in e.test.batch(128):
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

if exists(f'{ADV_DIR}/{e.attr}_adv_train') and exists(f'{ADV_DIR}/{e.attr}_adv_test'):
    g_adv_train = pickle.load(open(f'{ADV_DIR}/{e.attr}_adv_train','rb'))
    g_adv_test = pickle.load(open(f'{ADV_DIR}/{e.attr}_adv_test','rb'))

else:
    g_adv_train, g_adv_test = [], []

    for x in tqdm(adv_train_x):
        g_adv_train.append(eval(e.attr)(model,x))

    for x in tqdm(adv_test_x):
        g_adv_test.append(eval(e.attr)(model,x))

    g_adv_train, g_adv_test = np.array(g_adv_train), np.array(g_adv_test)

    pickle.dump(g_adv_train, open(f'{ADV_DIR}/{e.attr}_adv_train','wb'))
    pickle.dump(g_adv_test, open(f'{ADV_DIR}/{e.attr}_adv_test','wb'))

eprint('done\n')


### plotting ###
eprint('plotting ... ')

# plot attributions
plot_results([x_train, g_train, x_test, g_test],
            captions=['x_train',f'{e.attr}_train','x_test',f'{e.attr}_test'],
            filename=f'{EXP_DIR}/attr_norm.png')

# plot attributions of adversarial samples
plot_results(imgs=[adv_train_x, g_adv_train, adv_test_x, g_adv_test],
            captions=['adv_train', f'{e.attr}_adv_train', 'adv_test', f'{e.attr}_adv_test'],
            filename=f'{EXP_DIR}/attr_adv.png')

eprint('done\n')


### reconstruction ###
eprint('evaluating ...\n')

N_LABELS = 10

xc_train = gen_labels(model, e.train)
xc_test = gen_labels(model, e.test)

cls_train = np.array(list(xc_train.map(get_y, num_parallel_calls=tf.data.AUTOTUNE).as_numpy_iterator()))
cls_test = np.array(list(xc_test.map(get_y, num_parallel_calls=tf.data.AUTOTUNE).as_numpy_iterator()))

for t_label in range(N_LABELS):

    ### filtering ###
    tar_train = np.logical_and(y_train == cls_train, y_train == t_label)
    tar_test = np.logical_and(y_test == cls_test, y_test == t_label)

    cond_train = np.logical_and(adv_train_y == t_label, y_train != t_label)
    cond_test = np.logical_and(adv_test_y == t_label, y_test != t_label)

    g_adv_train_t = g_adv_train[cond_train]
    g_adv_test_t = g_adv_test[cond_test]

    g_train_t = g_train[tar_train]
    g_test_t = g_test[tar_test]

    ### train autoencoder for reconstruction ###
    eprint('configuring autoencoder ... ')

    ae = train_recons(eval(e.recons)(g_train[0].shape, latent_dim),
                      train=g_train_t,
                      test=g_test_t,
                      dir=f'{RECONS_DIR}/{e.recons}_{e.attr}_{t_label}',
                      bat_size=bat_size)

    eprint('done\n')

    ### evaluation ###
    eprint(f'... label {t_label} ... ')
    loss_fn = lambda x,y: lp_loss_q(x,y,order=order)

    # predictions
    r_train = ae.predict(g_train_t)
    r_test = ae.predict(g_test_t)
    r_adv_train = ae.predict(g_adv_train_t)
    r_adv_test = ae.predict(g_adv_test_t)

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
    pred_norm_tr = detect(ae, g_train_t, thresh, loss_fn)
    pred_norm_te = detect(ae, g_test_t, thresh, loss_fn)
    pred_anom_tr = detect(ae, g_adv_train_t, thresh, loss_fn)
    pred_anom_te = detect(ae, g_adv_test_t, thresh, loss_fn)

    # prediction and ground truth
    y_score = np.concatenate((test_loss, train_loss_adv, test_loss_adv))
    y_pred = np.concatenate((pred_norm_te, pred_anom_tr, pred_anom_te))
    y_true = np.concatenate((np.zeros_like(pred_norm_te), np.ones_like(pred_anom_tr), np.ones_like(pred_anom_te)))

    # get stats
    exp_conf = [e.target, e.recons, e.attr, e.attack, e.epsilon, order, t_label]
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
                captions=[f'{e.attr}_train', f'recons_train',
                          f'{e.attr}_test', f'recons_test',
                          f'{e.attr}_adv_train', f'recons_adv_train',
                          f'{e.attr}_adv_test', f'recons_adv_test'],
                filename=f'{EXP_DIR}/recons_{t_label}.png')

    eprint('done\n')

eprint('done\n\n')