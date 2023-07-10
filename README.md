# CIFAR-10 Image Classification

This repository contains scripts for training, evaluating, and using a pre-trained model for image classification on the CIFAR-10 dataset.

## Dataset Loader (cifar10_loader.py)

The `cifar10_loader.py` script downloads and prepares the CIFAR-10 dataset for training and evaluation. It loads the CIFAR-10 training and testing datasets, applies appropriate data transformations, and creates data loaders for convenient batch processing.

## Training Script (cifar10_train.py)

The `cifar10_train.py` script trains a ResNet-18 model on the CIFAR-10 dataset. It loads the training and validation datasets, applies data transformations, defines the model architecture, loss function, and optimizer. The script then performs a training loop over a specified number of epochs, printing the training progress and saving the trained model weights. Finally, it evaluates the model's accuracy on the test set.

## Evaluation Script (cifar10_eval.py)

The `cifar10_eval.py` script evaluates a pre-trained ResNet-18 model on the CIFAR-10 test set. It loads the test dataset, applies the same data transformations used during training, loads the pre-trained model, and performs inference on the test set. The script calculates the accuracy of the model's predictions and prints the result.

## Usage

1. Install the required dependencies by running the command: `pip install -r requirements.txt`.

2. Run the `cifar10_loader.py` script to download and prepare the CIFAR-10 dataset.

3. Train the model by running the `cifar10_train.py` script. Adjust the training parameters, such as the number of epochs, learning rate, etc., as needed.

4. Evaluate the trained model on the test set using the `cifar10_eval.py` script. Make sure to provide the path to the saved model weights (`resnet_model.pth`).

For more details on each script and customization options, please refer to the script files themselves.

## License

This project is licensed under the [MIT License](LICENSE).
