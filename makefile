### non-controllable arguments
ATTR := gradient	# attribution methods
RECONS := ae		# detector network

### controllable arguments
MODEL := cnn		# target model
DATA := cifar10		# target dataset
ATTACK := fgsm		# type of adversarial attacks
EPS := 0.1			# severity of adversarial attack
GPU := 0			# gpus to use
SEED := 0			# random seed

EXP_ARGS := $(MODEL) --data $(DATA) --attack $(ATTACK) --epsilon $(EPS) --attr $(ATTR) --recons $(RECONS) --gpu $(GPU) --seed $(SEED)

all: *.py
	python3 -u attr-adv.py $(EXP_ARGS)