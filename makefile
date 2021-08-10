### default arguments
MODEL := resnet		# target model
ATTR := gradient	# attribution methods
RECONS := cae		# detector network
DATA := cifar10		# target dataset
ATTACK := bim		# type of adversarial attacks
EPS := 0.03			# severity of adversarial attack
GPU := 0			# gpus to use
SEED := 0			# random seed

EXP_ARGS := $(MODEL) --data $(DATA) --attack $(ATTACK) --epsilon $(EPS) --attr $(ATTR) --recons $(RECONS) --gpu $(GPU) --seed $(SEED)

all: *.py
	python3 -u attr-adv.py $(EXP_ARGS)

gen-adv: *.py
	python3 -u gen-adv.py $(EXP_ARGS)