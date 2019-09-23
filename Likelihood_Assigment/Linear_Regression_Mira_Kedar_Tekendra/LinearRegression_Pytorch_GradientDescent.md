# 2. How is linear regression related to Pytorch and gradient descent?

Linear Regression is an approach that tries to find a linear relationship between a  **dependent variable**  and an  **independent variable**  by minimizing the distance as shown below.

![](https://miro.medium.com/max/60/1*F4JzgiTIUfFePLBj4A_JPw.jpeg?q=20)

![](https://miro.medium.com/max/700/1*F4JzgiTIUfFePLBj4A_JPw.jpeg)

Taken from  [https://www.youtube.com/watch?v=zPG4NjIkCjc](https://www.youtube.com/watch?v=zPG4NjIkCjc)

In this post, I’ll show how to implement a simple linear regression model  **using PyTorch.**

Let’s consider a very basic linear equation i.e.,  **y=2x+1**. Here, ‘x’ is the independent variable and y is the dependent variable. We’ll use this equation to create a dummy dataset which will be used to train this linear regression model. Following is the code for creating the dataset.

import numpy as np_# create dummy data for training  
_x_values = [i **for** i **in** range(11)]  
x_train = np.array(x_values, dtype=np.float32)  
x_train = x_train.reshape(-1, 1)  
  
y_values = [2*i + 1 **for** i **in** x_values]  
y_train = np.array(y_values, dtype=np.float32)  
y_train = y_train.reshape(-1, 1)

Once we have created the dataset, we can start writing the code for our model. First thing will be to define the model architecture. We do that using the following piece of code.

import torch  
from torch.autograd import Variable**class** linearRegression(torch.nn.Module):  
    **def** __init__(self, inputSize, outputSize):  
        super(linearRegression, self).__init__()  
        self.linear = torch.nn.Linear(inputSize, outputSize)  
  
    **def** forward(self, x):  
        out = self.linear(x)  
        **return** out

We defined a class for linear regression, that inherits  **torch.nn.Module**which is the basic Neural Network module containing all the required functions. Our Linear Regression model only contains one simple linear function.

Next, we instantiate the model using the following code.

inputDim = 1        _# takes variable 'x'_ outputDim = 1       _# takes variable 'y'  
_learningRate = 0.01   
epochs = 100  
  
model = linearRegression(inputDim, outputDim)  
_##### For GPU #######  
_**if** torch.cuda.is_available():  
    model.cuda()

After that, we initialize the loss (**Mean Squared Error**) and optimization (**Stochastic Gradient Descent**) functions that we’ll use in the training of this model.

criterion = torch.nn.MSELoss()   
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

After completing all the initializations, we can now begin to train our model. Following is the code for training the model.

**for** epoch **in** range(epochs):  
    _# Converting inputs and labels to Variable_ **if** torch.cuda.is_available():  
        inputs = Variable(torch.from_numpy(x_train).cuda())  
        labels = Variable(torch.from_numpy(y_train).cuda())  
    **else**:  
        inputs = Variable(torch.from_numpy(x_train))  
        labels = Variable(torch.from_numpy(y_train))  
  
    _# Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients_ optimizer.zero_grad()  
  
    _# get output from the model, given the inputs_ outputs = model(inputs)  
  
    _# get loss for the predicted output_ loss = criterion(outputs, labels)  
    print(loss)  
    _# get gradients w.r.t to parameters_ loss.backward()  
  
    _# update parameters_ optimizer.step()  
  
    print(**'epoch {}, loss {}'**.format(epoch, loss.item()))

Now that our Linear Regression Model is trained, let’s test it. Since it’s a very trivial model, we’ll test this on our existing dataset and also plot to see the original vs the predicted outputs.

**with** torch.no_grad(): _# we don't need gradients in the testing phase_ **if** torch.cuda.is_available():  
        predicted = model(Variable(torch.from_numpy(x_train).cuda())).cpu().data.numpy()  
    **else**:  
        predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()  
    print(predicted)  
  
plt.clf()  
plt.plot(x_train, y_train, **'go'**, label=**'True data'**, alpha=0.5)  
plt.plot(x_train, predicted, **'--'**, label=**'Predictions'**, alpha=0.5)  
plt.legend(loc=**'best'**)  
plt.show()

This plots the following graph.

![](https://miro.medium.com/max/60/1*JdLWW-oqZ_yAVTg0qJACNA.png?q=20)

![](https://miro.medium.com/max/640/1*JdLWW-oqZ_yAVTg0qJACNA.png)

Looks like our model has correctly figured out the linear relation between our dependent and independent variables.

If you have understood this, you should try and train a linear regression model for a little more  **complex linear equation with multiple independent variables**

**Gradient Descent Algorithm**

Gradient descent algorithm’s main objective is to minimise the cost function. It is one of the best optimisation algorithms to minimise errors (difference of actual value and predicted value).

Let’s represent the hypothesis h, which is function or a learning algorithm.

![](https://miro.medium.com/max/60/1*K1-0bnoMqxSv6tZXpOki1Q.png?q=20)

![](https://miro.medium.com/max/200/1*K1-0bnoMqxSv6tZXpOki1Q.png)

The goal is similar like the above operation that we did to find out a best fit of intercept line ‘y’ in the slope ‘m’. Using Gradient descent algorithm also, we will figure out a minimal cost function by applying various parameters for theta 0 and theta 1 and see the slope intercept until it reaches convergence.

In a real world example, it is similar to find out a best direction to take a step downhill.

![](https://miro.medium.com/max/60/1*09kq2L23D9XM_9Xtr8gc8Q.png?q=20)

![](https://miro.medium.com/max/700/1*09kq2L23D9XM_9Xtr8gc8Q.png)

Figure 3: Gradient Descent 3D diagram. Source:  [Coursera](https://www.coursera.org/learn/machine-learning/lecture/8SpIM/gradient-descent)  — Andrew Ng

We take a step towards the direction to get down. From the each step, you look out the direction again to get down faster and downhill quickly. The similar approach is using in this algorithm to minimise cost function.

We can measure the accuracy of our hypothesis function by using a cost function and the formula is

![](https://miro.medium.com/max/60/1*O0VceUdYADRY3FoTbBrqMA.png?q=20)

![](https://miro.medium.com/max/248/1*O0VceUdYADRY3FoTbBrqMA.png)

**_Gradient Descent for Linear Regression_**

![](https://miro.medium.com/max/60/1*S6sc4ohsd5ckBDY42QhivA.png?q=20)

![](https://miro.medium.com/max/300/1*S6sc4ohsd5ckBDY42QhivA.png)

Why do we use partial derivative in the equation? Partial derivatives represents the rate of change of the functions as the variable change. In our case we change values for theta 0 and theta 1 and identifies the rate of change. To apply rate of change values for theta 0 and theta 1, the below are the equations for theta 0 and theta 1 to apply it on each epoch.

![](https://miro.medium.com/max/60/1*TrFUjJR65rWBe3wN9Snacg.png?q=20)

![](https://miro.medium.com/max/300/1*TrFUjJR65rWBe3wN9Snacg.png)

To find the best minimum, repeat steps to apply various values for theta 0 and theta 1. In other words, repeat steps until convergence.

![](https://miro.medium.com/max/60/1*G3evFxIAlDchOx5Wl7bV5g.png?q=20)

![](https://miro.medium.com/max/450/1*G3evFxIAlDchOx5Wl7bV5g.png)

![](https://miro.medium.com/max/60/1*HrFZV7pKPcc5dzLaWvngtQ.png?q=20)

![](https://miro.medium.com/max/700/1*HrFZV7pKPcc5dzLaWvngtQ.png)

Figure 4: Gradient Descent for linear regression. Source:  [http://sebastianraschka.com/](http://sebastianraschka.com/)  — Python Machine Learning, 2nd Edition

-   where alpha (a) is a learning rate / how big a step take to downhill.
