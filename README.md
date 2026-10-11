Artificial Neural Network From Scratch using Numpy

About the Project

This is my implementation of a simple Artificial Neural Network (ANN) using Numpy. I made this project to understand how a neural network works internally instead of using a ready-made deep learning library

I used the iris dataset to train the model and predict the type of flower based on its measurements

Dataset

The iris dataset contains measurements of three types of flowers:

1 Setosa 2 Versicolor 3 Virginica

Each flower has four features: sepal length, sepal width , petal lengthand petal width

Data Preprocessing

I divided the dataset into three parts:

70% for training 15% for validation 15% for testing

I used stratified splitting so that the flower classes are represented in each part I also standardized the features using the mean and standard deviation of the training data. the target labels were converted into one hot encoded format.

Neural Network

The Network has four input features, two hidden layers with 8 and 6 neurons , and an output layer with 3 neurons

I implemented the main operations using Numpy incuding:

Weight and bias initialization forward propagation sigmoid activation for hidden layers softmax activation for the output layer cross-entropy loss backpropagation parameter updates using gradient descent

I also mini-batches and early stopping during training.

Results : The Neural Network was evaluted on 23 test samples from the iris dataset.

test accuracy: 100% (23 out of 23 correct predictions) setosa 7 out of 7 correctly classified versicolor 8 out of 8 correctly classified virginica 8 out of 8 correctly classfied

The confusion Matrix was

Actual Class Predicted Setosa Versicolor Virginica setosa 7 0 0

Versicolor 0 8 0

Virginica 0 0 8

The Validation accuracy reached 95.45% during training and was 90.91% at epoch 1000. the final ssaved model achived 100 % accuracy on this particular test split, since the test set contains only 23 samples, this result should not be interpreted as a gurantee of the same performance on unseen datasets

Experiment Comparison : Experiments compared learning rates of 0.001, 0.01, 0.1; sigmoid, ReLU and tanh hidden layer activations and architectures with hidden layers of 8-6 and 16-8 neurons

The best validation accuracy observed was 90.91% some configuration with this accuracy has substantially different validations losses, showing why both classification accuracy and loss should be considered the detailed results are available in experiment_results.csv

Visualizations

The Visualizations folder contains the plots created during the project including

Sepal and petal feature plots iris pairplot correlation heatmap training and validation loss graph validation accuracy graph

Project Files

iris_eda.py = explores the dataset and creates visualizations data_preprocessing.py = splits and preprocesses the data neural_network.py = contains the neural network functions train.py = trains the model and saves the training history evaluate.py = evaluates the saved model on test data best_model.npz = stores the trained model parameters training_history.csv = stores loss and accuracy values for each epoch requirements.txt = lists the python package used in the environment decision_boundary.py = creates a PCA based graph showing how the neural network separates the three iris flower classes sklearn_comparison.py = compares the Numpy neural networks test accuracy with Scikit-learns MLPClassifier run_experiments.py = tests 18 combinations of learning rates , activation functions, and network acrhitectures , saving the results to experiment_results.csv

How to run

first , activate the virtual environent

source .venv/bin/activate

install the required packages:

pip install -r requirements.txt

Run the scripts in this order

python iris_eda.py python train.py python evaluate.py python decision_boundary.py python sklearn_comparison.py python run_experiments.py

Conclusion :

Through this project , I got practical experience with the basic working of an artificial neural network . Implementing forward propagation , backpropagation and gradient descent helped me understand how the model learns from data using Numpy
