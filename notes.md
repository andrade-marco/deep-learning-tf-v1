# Tensorflow for Deep Learning

#### ML workflow
1. Data acquisition
2. Data cleaning
3. Train/Test split
    - Unsupervised learning: generally do not do it for unsupervised learning since 
      we do not know the answer for what we are looking for
    - This can also be split into: train, test, holdout (data model truly did not see)
4. Train model
5. Evaluate model - use test set
6. Adjust model parameters
7. Re-train (repeat 5, 6, 7 until ready)
8. Deploy model

### Neural Network
- `z = w*x + b --> a(z)`

##### Activation functions (a)
  - Tanh
  - Sigmoid
  - ReLu
##### Cost functions
Use a cost function to measure how far off we are
  from real value
  - *Quadratic cost* --> `C = Sum(y - y_hat)^2 / n`
    - Larger errors are more prominent due to the squaring
    - It slows down calculation
  - *Cross Entropy* --> `C = (-1/n)*Sum(yln(y_hat)+(1-y)ln(1-y_hat))`
    - Faster learning
    - The larger the difference, the faster the neuron can learn

##### Gradient Descent
Gradient descent is an optimization algorithm for finding the minimum of a function. To find a local
minimum, we take steps proportional to the negative of the gradient. That's good for us, since we are
trying to find the minima for the cost function.

Backpropagation is used to calculate the error contribution of each neuron after a batch of data is
processed. It relies heavily on the chain rule to go back through the network and calculate these
errors. It works by calculating the error at the output and then distributes back through the network
layers. It requires a known desired output for each input value - supervised learning.

