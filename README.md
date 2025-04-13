# DDSE: A Decoupled Dual-Stream Enhanced Framework for Multimodal Sentiment Analysis with Text-Centric SSM

## Usage

### Prerequisites
Python version: 3.8.20

PyTorch version: 2.0.1

CUDA version: 11.7

### Dataset Settings and Default Settings
Data files (containing processed MOSI, MOSEI datasets) can be downloaded from [here](https://drive.google.com/drive/folders/1BBadVSptOe4h8TWchkhWZRLJw8YG_aEi?usp=sharing). 
You should put them in the `./dataset` folder, or modify the dataset path on your own in `config/config.json`. Before running, you need to set the necessary parameters in `./config/config.json`. Run logs and run results are saved in the `./log` and `./result/normal` directories by default, respectively.

### Run the Code
- Install the necessary packages
```
pip install -r requirements.txt
```

- Training

You can first set the dataset_name='mosi' or dataset_name='mosei' in `train.py`, and then run:
```
python train.py
```
By default, the trained model will be saved in the `./pt` directory. You can change this in `train.py`.
- Testing

You can first set the dataset_name='mosi' or dataset_name='mosei' in `test.py`, and set the path of the trained model in `run.py`. Then test the trained model:
```
python test.py
```

