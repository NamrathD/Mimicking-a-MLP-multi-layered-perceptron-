# Mimicking-a-MLP-multi-layered-perceptron-
I mimicked a MLP function to get more in-depth understanding of MLP and back-propagation.

Some lessons I learned along the way are:


•	How we initialize values for weights can impact how effectively the program computes. 
•	Unlike Linear regression, we can’t initialize all the weight values equaling to some constant. It is a good idea to initialize them to some random values.
•	Scaling and centering the data results in higher accuracy in output.
•	More layers requires more number of iterations to produce higher confident outputs
•	Creating too many layers than substantial amount, will lead to overfitting the training data, and increases the error cost for test data.
•	When too many layers causes overfitting, it can be addressed through increasing the regularization hyperparameter (which will try to underfit).
