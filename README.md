# CIFAR-10 Image Classification using Convolutional Neural Networks

This project implements a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset using PyTorch.

## Description
The goal of this project is to build, train, and evaluate a Convolutional Neural Network (CNN) to recognize and classify images from the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes. The classes include:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

The project demonstrates how to preprocess the data, define a CNN, train the network, evaluate its performance, and save/load the trained model.

## Repository Contents
- `cifar10-image-classification.ipynb`: Jupyter notebook with the project implementation, including data preprocessing, model definition, training, and evaluation.
- `requirements.txt`: List of dependencies required to run the notebook.

## Installation and Setup
To run this project, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/regmibhuwan/cifar10-image-classification.git

2. **Navigate to the Project Directory**:

  cd cifar10-image-classification

3. **Install the Dependencies**:

  pip install -r requirements.txt

4. **Open the Jupyter Notebook**:

  jupyter notebook cifar10-image-classification.ipynb

# Project Details
## Data Loading and Preprocessing
- Libraries Used: The project uses PyTorch and torchvision for handling image data and transformations.
- Transformations: Images are converted to tensors and normalized to have values between -1 and 1.
## Model Definition
- Convolutional Neural Network (CNN): The CNN is defined using PyTorch's nn.Module class.
- Layers: The network consists of two convolutional layers followed by max-pooling layers, and three fully connected layers.
- Activation Functions: ReLU activation functions are used after each convolutional and fully connected layer.
## Training the Model
- Loss Function: Cross-Entropy Loss is used for classification.
- Optimizer: Stochastic Gradient Descent (SGD) with momentum is used to update the model weights.
- Training Loop: The training loop iterates over the dataset multiple times (epochs), performing forward and backward passes to minimize the loss.
## Evaluating the Model
- Test Data: The model is evaluated on the CIFAR-10 test dataset.
- Accuracy Calculation: The accuracy of the model is calculated by comparing the predicted labels with the actual labels.
- Class-wise Accuracy: The accuracy for each class is also calculated to understand the model's performance on different categories.
## Saving and Loading the Model
- Saving: The trained model's state is saved to a file using torch.save().
- Loading: The saved model can be loaded back using torch.load() to make predictions or further evaluations.
## Results
The notebook includes detailed results of the training process, including loss values and accuracy metrics. Visualizations of sample images along with their predicted and actual labels are provided to demonstrate the model's performance.

## Usage
### Run the Jupyter Notebook:
- Open cifar10-image-classification.ipynb in Jupyter Notebook or Jupyter Lab.
### Execute the Cells:
- Follow the instructions in the notebook and execute the cells to train and evaluate the CNN model.
## License
- This project is licensed under the MIT License
## Acknowledgements
This project is part of the  PyTorch 60-Minute Deep Learning Blitz tutorial.
  
