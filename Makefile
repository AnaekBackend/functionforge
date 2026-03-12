.PHONY: train eval sweep

# Default config file
CONFIG ?= configs/default.yaml

train:
	python train.py --config $(CONFIG)

eval:
	python eval.py

predict:
	python predict.py "$(QUERY)"

sweep:
	@echo "Sweep not implemented yet"
