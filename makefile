### default arguments
MODEL := cnn		# target model
ATTR := gradient	# attribution methods
RECONS := ae		# detector network
DATA := mnist		# target dataset
ATTACK := pgd		# type of adversarial attacks
EPS := 0.3			# severity of adversarial attack
GPU := 0			# gpus to use
SEED := 0			# random seed

EXP_ARGS := $(MODEL) --data $(DATA) --attack $(ATTACK) --epsilon $(EPS) --attr $(ATTR) --recons $(RECONS) --gpu $(GPU) --seed $(SEED)

all: *.py
	python3 -u main.py $(EXP_ARGS)

test_adv: *.py
	python3 -u test_adv.py $(EXP_ARGS)