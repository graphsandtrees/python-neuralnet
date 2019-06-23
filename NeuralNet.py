import random
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


    def symbolic_error_derivative(self, x, target, i, j): #not agreeing with numerical
        error_derivative = -1*x[i]*(target[j]-self.compute_output(x)[j])
        return error_derivative

    def learn_method_1(self, x, target):
        error = 100
        iter = 0
        while error >= 1:
            output = self.compute_output(x)
            self.weights
            for i in range(self.x_length):
                for j in range(self.output_length):
                    self.weights[j][i] -= self.numeric_derivative_error_right(x, target, i, j, compute_error(x, target))*self.weights[j][i]
                    debug(error)
                print(i)
            error = self.compute_error(self.compute_output(x), target)
            iter += 1
    def learn_method_2(self, x, target):
        error = 100
        iter = 0
        while error >= 1:
            output = self.compute_output(x)
            self.weights
            for i in range(self.x_length):
                for j in range(self.output_length):
                    self.weights[j][i] -= self.symbolic_error_derivative(x, target, i, j, self.compute_error(x, target))*self.weights[j][i]
                    debug(error)
                print(i)
            error = self.compute_error(self.compute_output(x), target)
            iter += 1
	
        return error
        return self.compute_output(x)
def debug(*x):
    return
    for i in x:
        print(i)

def to_one_hot(i, length):
    return [0]*i+[1]+[0]*(length-i-1)    

assert to_one_hot(3,10) == [0,0,0,1,0,0,0,0,0,0]

def main(): 
    x = images[0]
    target = to_one_hot(label[0])
    print labels[0]
    a = neuralnet(len(x), len(target))

    x = [random.random() for i in range(n)] #784 can be replaced by len(x), 10 by output_length
    #output = a.compute_output(x)
    #error = a.compute_error(output, target)
    #error_derivative_numeric = a.numeric_derivative_error(x, target, 0, 0)
    #error_derivative_symbolic = a.symbolic_error_derivative(x, target, 0, 0)
    #debug("error: ",error)
    #debug("output 1: ", output)
    #debug("numeric: ",error_derivative_numeric)
    #debug("symbolic: ",error_derivative_symbolic)
    #error = a.learn_method_1(x, target)
    #debug("error 2:", error)
    #a.learn_method_2(x, target)

    #debug("output 2:", output)
    
main()


