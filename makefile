### non-controllable arguments
ATTR := grad		# attribution methods
RECONS := ae		# detector network

### controllable arguments
MODEL := vgg16		# target model
DATA := cifar10		# target dataset
ATTACK := fgsm		# type of adversarial attacks
EPS := 0.1			# severity of adversarial attack
GPU := 0			# gpus to use

EXP_ARGS := $(MODEL) --data $(DATA) --attack $(ATTACK) --epsilon $(EPS) --attr $(ATTR) --recons $(RECONS) --gpu $(GPU)

all: *.py
	python3 -u attr-adv.py $(EXP_ARGS)