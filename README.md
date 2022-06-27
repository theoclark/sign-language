# Sign-Language Detection App

App to read the American Sign Language alphabet. For more information, read the [blog post](https://theoclark.co.uk/posts/sign-language).

![example](https://github.com/theoclark/sign_language/blob/main/example.gif)

## Usage

To use the app, download or clone the repository and run ```bash python app.py```. The model works best against a clear background. The hand must be inside the box.

American Sign Language alphabet:

![alphabet](https://github.com/theoclark/sign_language/blob/main/Symbols.png)

## Training

Use the train.ipynb notebook to train a model from scratch.

A similar training set of 28x28 grayscale images is available via [Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) and this was used initially for training. However, the lack of variation within the dataset meant that the model didn't generalise well to different backgrounds and situations.

I created a new dataset of 256x256 colour images containing 3279 training images and 1069 test images. The images contain a variety of backgrounds, lighting, hand positions and angles. The dataset can be downloaded [here](https://drive.google.com/file/d/1I35bpJ4ck3nDT1SzEs-6DqmJR7CtcA3Q/view?usp=sharing).

The model is a 12 block ConvNet built using PyTorch. Each block consists of a 3x3 2D convolution layer, a Tanh activation layer and a batch-norm layer. The first convolution is 5x5. The channels dimensions are: 6, 16, 120, 120, 120, 120, 120, 120, 120, 120, 64, 26. For more details, see the source code.
