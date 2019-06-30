import random
import pdb
from mnist import MNIST
mndata = MNIST('./python-mnist/data')
images, labels = mndata.load_training()


class neuralnet():
    def __init__(self, x_length, output_length):
        self.weights = [[random.random() for i in range(x_length)] for j in range(output_length)] #784 can be replaced by len(x), 10 by output_length
        self.x_length = x_length
        self.output_length = output_length

    def compute_output(self, x):        #computes output based on weights and x
        output = [0]*self.output_length
        for i in range(self.x_length):
            for j in range(self.output_length):
                output[j] += x[i]*self.weights[j][i]
        return output

    def compute_error(self, output, target):
        assert len(output) == self.output_length
        error = 0
        for j in range(self.output_length):
            error += ((target[j] - output[j])**2)/2
        return error

    def numeric_derivative_error(self, x, target, i, j):
        delta = .0001
        self.weights[j][i]+=delta
        error = (self.compute_error(self.compute_output(x), target))
        self.weights[j][i]-=2*delta
        error_derivative = (error - (self.compute_error(self.compute_output(x), target)))/(2*delta)
        self.weights[j][i]+=delta
        return error_derivative


    def numeric_derivative_error_right(self, x, target, i, j, error):
        delta = .0001
        self.weights[j][i]+=delta
        new_output = self.compute_output(x)
        new_error = self.compute_error(new_output, target)
        error_derivative = (error - new_error)/(2*delta)
        self.weights[j][i]-=delta
        return error_derivative


    def symbolic_error_derivative(self, x, target, i, j): 
        error_derivative = -1*x[i]*(target[j]-self.compute_output(x)[j])
        return error_derivative

    def learn(self, x, target):
        output = self.compute_output(x)
        #pdb.set_trace()
        for i in range(self.x_length):
            for j in range(self.output_length):
                self.weights[j][i] -= self.symbolic_error_derivative(x, target, i, j)*self.weights[j][i]
	
    def learnmnist(self, images,labels, number_to_learn):
        for image, label in zip(images[:number_to_learn],labels[:number_to_learn]):
            target = to_one_hot(label,10)
            print "before", self.compute_error(self.compute_output(image), target)
            self.learn(image, target)
            print "after", self.compute_error(self.compute_output(image), target)

def debug(*x):
    return
    for i in x:
        print(i)

def to_one_hot(i, length):
    return [0]*i+[1]+[0]*(length-i-1)    

assert to_one_hot(3,10) == [0,0,0,1,0,0,0,0,0,0]

def main(): 
    x = images[0]
    target = to_one_hot(labels[0], 10)
    print labels[0]
    test_net = neuralnet(len(x), len(target)) 
    test_net.learnmnist(images, labels, 10)
    
main()
