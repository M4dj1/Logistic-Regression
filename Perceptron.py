import numpy as np
import matplotlib.pyplot as plt
pointsnum=300
learning_rate = 0.01

def labeling(array) :
    for i in range(pointsnum) :
        if array[i][0] + array[i][1] - 1 > 0 :  # (x1 + x2 - 1 > 0)
            array[i][2] = 1
        else :
            array[i][2] = -1

class perceptron :

    def __init__(self) :
        self.weights = np.random.random_sample((1,2))
        self.bias = 0.5

    def output(self,input) :
        if (self.weights.dot(input)+self.bias)>0  :
            return 1
        else :
            return -1

    def update(self,input, error):
        self.weights = self.weights + np.matrix.transpose(learning_rate*(error)*input)
        self.bias = self.bias + learning_rate*error

#Initializing dataset
randomPoints = np.random.uniform(-5,5,[pointsnum, 3])
labeling(randomPoints)
#End


p = perceptron()


err=0
CounterErrors = 0
i=0
while i<pointsnum :
    inputs = np.matrix.transpose(np.copy([[randomPoints[i][0],randomPoints[i][1]]]))
    output = p.output(inputs)
    plt.scatter(inputs[0], inputs[1], c='r')
    if output != randomPoints[i][2] :
        err = randomPoints[i][2] - output
        CounterErrors+=1
        p.update(inputs, err)
    i += 1
print("Number Of Errors", CounterErrors)

#Show Data
x = np.linspace(-5, 5, 100)
y = -(p.weights[0][0] / p.weights[0][1]) * x - (p.bias / p.weights[0][1])  # (x1(w1) + x2(w2) - 1(w0) = 0) ==> x1 = -(w1/w2)x2 - (bias/w2)
plt.plot(x, y)
plt.show()
