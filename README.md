# Unsupervised Detection of Adversarial Examples with Model Explanations, AdvML @ KDD'21

**Gihyuk Ko<sup>1,2</sup>, Gyumin Lim<sup>2</sup>**

<sup>1</sup> Carnegie Mellon University\
<sup>2</sup> Cyber Security Research Center, Korea Advanced Institute of Science and Technology (KAIST)

---

## Installation

```bash
$ git clone https://github.com/gihyukko/tool-attr-adv.git
$ cd tool-attr-adv
```

### Environment
  * Python 3.8+
  * TensorFlow 2.4+

The base environment be set up as:
```bash
$ pip install -r requirements
```

### Datasets

MNIST dataset used in the paper is automatically downloaded to ```./data``` directory when running the experiments for the first time.

### Running the script

The script can be run with the following command:
```bash
$ make
```

The configuration for experiments is provided primarily via command-line arguments, where their default values are set in [makefile](makefile). Currently, only limited settings are controllable. The following is an example configuration for controllable arguments.

```bash
make GPU=1,2 ATTACK=mim EPS=0.5
```

Running the script will automatically train and save trained models to ```experiments``` directory. Figures will be generated and stored in ```figures```, and quantitative results will be stored in ```bench_all.tsv```.

## Contact

In case of feedback, suggestions, or issues, please contact [Gihyuk Ko](https://gihyukko.github.io/).