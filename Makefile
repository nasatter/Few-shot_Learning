# OS detect from https://gist.github.com/sighingnow/deee806603ec9274fd47
ARCH := ''
# TODO add gpu flag?
OSFLAG :=
ifeq ($(OS),Windows_NT)
	OSFLAG=WIN32
	ifeq ($(PROCESSOR_ARCHITECTURE),AMD64)
		ARCH=AMD64
	endif
	ifeq ($(PROCESSOR_ARCHITECTURE),x86)
		ARCH=IA32
	endif
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		OSFLAG=LINUX
	endif
	ifeq ($(UNAME_S),Darwin)
		OSFLAG=OSX
	endif
		UNAME_P := $(shell uname -p)
	ifeq ($(UNAME_P),x86_64)
		ARCH=AMD64
	endif
		ifneq ($(filter %86,$(UNAME_P)),)
	ARCH=IA32
		endif
	ifneq ($(filter arm%,$(UNAME_P)),)
		ARCH=ARM
	endif
endif

# checks if in python venv
#

init: ## Sets up and switches to python venv
	$(eval IN_VENV=$(python -c 'import sys; print (1 if sys.prefix == sys.base_prefix else 0)'))
	@echo INVENV $(IN_VENV)

show_os_arch: ## Prints current os and cpu arch
	@echo OS:$(OSFLAG) ARCH:$(ARCH)

run: ## Run models
ifeq ($(OSFLAG), WIN32)
	./run.bat
endif
ifeq ($(OSFLAG),LINUX)
	./run.sh
endif
ifeq ($(OSFLAG),OSX)
	./run.sh
endif

#debug:
	#TODO debug command

lint: ## Lint the code
	black .
	isort .
	flake8 /srv/app/ckanext/ --count --max-line-length=127 --show-source --statistics --exclude ckan

#test: ## Run tests in a new container
	# TODO run tests coammand


.DEFAULT_GOAL := help
.PHONY: build clean help lint test up

# Output documentation for top-level targets
# Thanks to https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
help: ## This help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-10s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
