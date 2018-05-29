# ASL Alphabet Translator

This project is a sign language alphabet recognizer using Python, openCV and TensorFlow for training InceptionV3 model, a convolutional neural network model for classification.


## Requirements

This project uses python 3.6 and the following packages:
* opencv
* tensorflow
* matplotlib
* numpy

```
## Training

To train the model, use the following command:
```
python3 train.py \
  --bottleneck_dir=logs/bottlenecks \
  --how_many_training_steps=2000 \
  --model_dir=inception \
  --summaries_dir=logs/training_summaries/basic \
  --output_graph=logs/output_graph.pb \
  --output_labels=logs/output_labels.txt \
  --image_dir=./dataset
```
  
## Classifying
  
To test classification, use the following command:
```
python3 classify.py path/to/image.jpg
```

## Using webcam (demo)

To use webcam, use the following command:
```
python3 classify_video.py
```
Your hand must be inside the box.
