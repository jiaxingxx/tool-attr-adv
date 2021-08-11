# utilities
from util import *
from ml_util import *

# import custom modules
from attacks import *
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
                      {'eps': e.epsilon, 'clip_min':0.0, 'clip_max':1.0}],
            'spsa'  : ['spsa',
                      {'eps': e.epsilon, 'nb_iter': 40, 'clip_min':0.0, 'clip_max':1.0}]}

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
for t_label in range(1):

    eprint(f'generating adversarial examples for {t_label} ... ')

    if exists(f'{ADV_DIR}/adv_train_{t_label}') and exists(f'{ADV_DIR}/adv_test_{t_label}'):
        (adv_train_x, adv_train_y) = pickle.load(open(f'{ADV_DIR}/adv_train_{t_label}','rb'))
        (adv_test_x, adv_test_y) = pickle.load(open(f'{ADV_DIR}/adv_test_{t_label}','rb'))

    else:
        adv_train_x, adv_train_y = [], []
        adv_test_x, adv_test_y = [], []
        import time

        for x,y in e.train.batch(1):
            c = time.time()
            atk_args['y'] = y
            x_adv = eval(atk_method)(model, x, **atk_args)
            e = time.time()
            y_adv = np.argmax(model.predict(x_adv), axis=1)
            adv_train_x.append(x_adv)
            adv_train_y.append(y_adv)

            print(f'{e-c} seconds')

            plot_img(x[0], filename='original.png')
            print(y[0])
            plot_img(x_adv[0], filename='adversarial.png')
            print(y_adv[0])

            exit()

        adv_train_x = tf.concat(adv_train_x, axis=0)
        adv_train_y = tf.concat(adv_train_y, axis=0)

        for x,_ in e.test.batch(128):
            atk_args['y'] = [t_label]*len(x)
            x_adv = eval(atk_method)(model, x, **atk_args)
            y_adv = np.argmax(model.predict(x_adv), axis=1)
            adv_test_x.append(x_adv)
            adv_test_y.append(y_adv)

        adv_test_x = tf.concat(adv_test_x, axis=0)
        adv_test_y = tf.concat(adv_test_y, axis=0)

        pickle.dump((adv_train_x, adv_train_y), open(f'{ADV_DIR}/adv_train_{t_label}','wb'))
        pickle.dump((adv_test_x, adv_test_y), open(f'{ADV_DIR}/adv_test_{t_label}','wb'))

eprint('done\n')
