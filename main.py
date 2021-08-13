# utilities
from util import *
from ml_util import *

# import custom modules
from attacks import *
from attribution import *
from models import *
from gen_data import *

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
                      {'eps': e.epsilon, 'clip_min':0.0, 'clip_max':1.0}],
            'spsa'  : ['spsa',
                      {'eps': e.epsilon, 'nb_iter': 40, 'clip_min':0.0, 'clip_max':1.0}]}

atk_method, atk_args = atk_dict[e.attack]

# parameters for training and evaluation
bat_size = 32
latent_dim = 64
order = 1.0

# directories
MODEL_DIR = f'experiments/{e.target}'
RECONS_DIR = f'{MODEL_DIR}/recons'
DATA_DIR = f'{MODEL_DIR}/data'
ATTR_DIR = f'{DATA_DIR}/{e.attr}'
ADV_DIR = f'{DATA_DIR}/{e.attack}_{e.epsilon}'
FIG_DIR = f'figures/{e.target}_{e.recons}_{e.attr}_{e.attack}_{e.epsilon}'

# make directories if path does not exist
dir_names = ['experiments', MODEL_DIR, RECONS_DIR, DATA_DIR, ATTR_DIR, ADV_DIR, 'figures', FIG_DIR]
mkdir(dir_names)

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
g_train = gen_attr(model, e.train, eval(e.attr), e.attr_spec, f'{ATTR_DIR}/train')
g_test = gen_attr(model, e.test, eval(e.attr), e.attr_spec, f'{ATTR_DIR}/test')
eprint('done\n')

### generate adversarial examples ###
eprint('generating adversarial examples ... ')
ae_train = gen_ae(model, e.train, eval(atk_method), atk_args, f'{ADV_DIR}/ae_train')
ae_test = gen_ae(model, e.test, eval(atk_method), atk_args, f'{ADV_DIR}/ae_test')
eprint('done\n')

### generate explanations for adversarial images ###
eprint('generating attributions for adversarial examples ...\n')
g_ae_train = gen_attr(model, ae_train,
                      attr_fn=eval(e.attr), attr_spec=e.attr_spec,
                      dir=f'{ADV_DIR}/{e.attr}_ae_train')
g_ae_test = gen_attr(model, ae_test,
                     attr_fn=eval(e.attr), attr_spec=e.attr_spec,
                     dir=f'{ADV_DIR}/{e.attr}_ae_test')
eprint('done\n')

### plotting ###
eprint('plotting ... ')

x_train = e.train.map(get_x, num_parallel_calls=tf.data.AUTOTUNE)
x_test = e.test.map(get_x, num_parallel_calls=tf.data.AUTOTUNE)
x_ae_train = ae_train.map(get_x, num_parallel_calls=tf.data.AUTOTUNE)
x_ae_test = ae_test.map(get_x, num_parallel_calls=tf.data.AUTOTUNE)

# plot attributions
plot_results([x_train, g_train, x_test, g_test],
            captions=['x_train',f'{e.attr}_train','x_test',f'{e.attr}_test'],
            filename=f'{FIG_DIR}/attr_norm.png')

# plot attributions of adversarial samples
plot_results(imgs=[x_ae_train, g_ae_train, x_ae_test, g_ae_test],
            captions=['ae_train', f'{e.attr}_ae_train', 'ae_test', f'{e.attr}_ae_test'],
            filename=f'{FIG_DIR}/attr_ae.png')

eprint('done\n')

### reconstruction ###
eprint('evaluating ...\n')

N_LABELS = 1

#xc_train = gen_labels(model, e.train)
#xc_test = gen_labels(model, e.test)

#cls_train = np.array(list(xc_train.map(get_y, num_parallel_calls=tf.data.AUTOTUNE).as_numpy_iterator()))
#cls_test = np.array(list(xc_test.map(get_y, num_parallel_calls=tf.data.AUTOTUNE).as_numpy_iterator()))

c_train = gen_pred(model, e.train)
c_test = gen_pred(model, e.test)

y_train = to_array(map_ds(e.train, get_y))
y_test = to_array(map_ds(e.test, get_y))

g_train = to_array(g_train)
g_test = to_array(g_test)

g_ae_train = to_array(g_ae_train)
g_ae_test = to_array(g_ae_test)

for t_label in range(N_LABELS):

    ### filtering ###
    tar_train = np.logical_and(y_train == c_train, y_train == t_label)
    tar_test = np.logical_and(y_test == c_test, y_test == t_label)

    cond_train = np.logical_and(ae_train_y == t_label, y_train != t_label)
    cond_test = np.logical_and(ae_test_y == t_label, y_test != t_label)

    g_ae_train_t = g_ae_train[cond_train]
    g_ae_test_t = g_ae_test[cond_test]

    g_train_t = g_train[tar_train]
    g_test_t = g_test[tar_test]

    ### train autoencoder for reconstruction ###
    eprint('configuring autoencoder ... ')

    ae = train_recons(eval(e.recons)(e.attr_shape, latent_dim),
                      train=g_train_t,
                      test=g_test_t,
                      dir=f'{RECONS_DIR}/{e.recons}_{e.attr}_{t_label}',
                      bat_size=bat_size)

    eprint('done\n')

    ### evaluation ###
    eprint(f'... label {t_label} ... ')
    loss_fn = lambda x,y: lp_loss_q(x,y,p=order,q=0.0)

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
                filename=f'{FIG_DIR}/recons_{t_label}.png')

    eprint('done\n')

eprint('done\n\n')