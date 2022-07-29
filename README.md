# AirQo Ugandan Air Quality Forecast

In this repo, we want to forecast, whether the air is so polluted, that outdoor activities should be cancelled or not. 

The data used for this is found here: [AirQo Ugandan Air Quality Forecast Challenge](https://zindi.africa/competitions/airqo-ugandan-air-quality-forecast-challenge).

---
## Requirements and Environment

Requirements:
- pyenv with Python: 3.9.8

Environment: 

For installing the virtual environment you can either use the Makefile and run `make setup` or install it manually with the following commands: 

```Bash
pyenv local 3.9.8
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

In order to train the model and store test and train data in the data folder and the model in models run:

```bash
python files/train.py  
```

In order to test that predict works on the test set you created run:

```bash
python files/predict.py
```
