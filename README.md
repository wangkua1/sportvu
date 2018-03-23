# sportvu
work in progress... browse at your own risk

# Pick and Roll Detection

### Setup

1. Download and extract the data used for the repository [here](https://drive.google.com/open?id=0B3s12MhYb3jOZFM4V1U2emdwQ0k).

2. Clone the repository
```
git clone https://github.com/wangkua1/sportvu/
```

3. Modify the `sportvu/data/constant.py` file to contain the path to the data folder extracted previously.

4. Install the sportvu package with the following commands.
```
sudo python setup.py build
sudo python setup.y install
```

5. Create annotation file from the data folder. In `sportvu/data`, run the `annotation.py` script.
```
python annotation.py
```

6. Create numpy files containing sportvu sequence information. In `sportvu`, run the `make_sequences_from_sportvu.py` file with a proper data config file. This should take approx one hour.
```
python make_sequences_from_sportvu.py rev3_1-bmf-25x25.yaml
```

7. Train a model on the prepared sequences. Specify a proper data and model config file.
```
python train.py 0 rev3_1-bmf-25x25.yaml conv2d-3layers-25x25.yaml
```

8. In order to observe precision recall curve, raw probabilities are written to file. Raw probs can be created from train and val splits. The two testing options can be ran below.
```
python train.py 0 rev3_1-bmf-25x25.yaml conv2d-3layers-25x25.yaml --test 5
python train.py 0 rev3_1-bmf-25x25.yaml conv2d-3layers-25x25.yaml --test --train 5
```

9. To view the ROC curve from the probabilities written to file, detection operations are performed. Run the detection for a precision-recall graph. Specify a model, data, and detection config file.
```
python detection_pr.py 0 rev3_1-bmf-25x25.yaml conv2d-3layers-25x25.yaml nms1.yaml 5 --single
```

10. To view the graphs of the probabilities during the duration of a movement sample, detection can be ran from a seperate script.
```
python detection_from_raw_pred.py 0 rev3_1-bmf-25x25.yaml conv2d-3layers-25x25.yaml nms1.yaml
```
