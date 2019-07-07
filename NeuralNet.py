import random
import pdb
from mnist import MNIST


def rescale_image(image):
    return [(i-127.5)/256 for i in image]

class neuralnet():
    def __init__(self, x_length, output_length, rate=.01):
        self.weights = [[random.uniform(-1,1) for i in range(x_length)] for j in range(output_length)] 
        self.x_length = x_length
        self.output_length = output_length
        self.learn_rate = rate

    def compute_output(self, x):        #computes output based on weights and x
        output = [0]*self.output_length
        for j in range(self.output_length):
            for i in range(self.x_length):
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
        error_derivative = float(-1*x[i]*(target[j]-self.compute_output(x)[j]))
        return error_derivative

    def learn(self, x, target):
        for j in range(self.output_length):
            for i in range(self.x_length):
                self.weights[j][i] -= self.learn_rate * self.symbolic_error_derivative(x,target,i,j)

    def learnmnist(self, images,labels, number_to_learn):
        for image, label in zip(images[:number_to_learn],labels[:number_to_learn]):
            target = to_one_hot(label,10)
            #pdb.set_trace()

            self.learn(image, target)
            print self.compute_error(self.compute_output(image), target)

def debug(*x):
    return
    for i in x:
        print(i)

def to_one_hot(i, length):
    return [0]*i+[1]+[0]*(length-i-1)    

assert to_one_hot(3,10) == [0,0,0,1,0,0,0,0,0,0]


def main(): 
    mndata = MNIST('./python-mnist/data')
    length_to_parse = 20
    pre_images, labels = mndata.load_training()
    images = map(rescale_image, pre_images[:length_to_parse])
    labels = labels[:length_to_parse]

    x = images[0]
    target = to_one_hot(labels[0], 10)
    print labels[0]
    test_net = neuralnet(len(x), len(target)) 
    print test_net.compute_error(test_net.compute_output(x), target)
    test_net.learnmnist(images, labels, 10)
    assert abs(test_net.numeric_derivative_error(x,target,152,3) - test_net.symbolic_error_derivative(x,target,152,3)) < abs(test_net.numeric_derivative_error(x,target,152,3)) * .02
    print test_net.compute_output(x)
main()
