# Context based OS fingerprinting - the code base

## Introduction
This repository compares three OS fingerprinting models, collects the results and checks if there is a statistical significant difference. It is the codebase for our project report.

## Data disclaimer
Since we respect the privacy of persons who provided the network data voluntarily we unfortunatly cannot publish them. If you still wish to receive them please e-mail us on kilian.merke@fau.de.


## Setup
#### Setting up virtual environment (Linux, MacOS)
```
python3 -m venv .venv
source .venv/bin/activate
```

#### Installing dependencies
```
python3 -m pip install -r requirements.txt
```

## Usage
#### Convert data
```
python3 data_preparation.py
```

#### Run experiment pipeline
```
python3 pipeline.py
```

#### Run evaluation
```
python3 evaluation.py
```