import random

class neuralnet():
    def __init__(self, input_length, output_length):
        self.weights = [[random.random() for i in range(input_length)] for j in range(output_length)] #784 can be replaced by len(input), 10 by output_length
        self.input_length = input_length
        self.output_length = output_length

    def compute_output(self, input):        #computes output based on weights and input
        output = [0]*self.output_length
        for i in range(self.input_length):
            for j in range(self.output_length):
                output[j] += input[i]*self.weights[j][i]
        return output

    def compute_error(self, output, target):
        assert len(output) == self.output_length
        error = 0
        for j in range(self.output_length):
            error += ((target[j] - output[j])**2)/2
        return error

    def numeric_derivative_error(self, input, target, i, j):
        delta = .0001
        self.weights[j][i]+=delta
        error = (self.compute_error(self.compute_output(input), target))
        self.weights[j][i]-=2*delta
        error_derivative = (error - (self.compute_error(self.compute_output(input), target)))/(2*delta)
        self.weights[j][i]+=delta
        return error_derivative


    def numeric_derivative_error_right(self, input, target, i, j, error):
        delta = .0001
        self.weights[j][i]+=delta
        new_output = self.compute_output(input)
        new_error = self.compute_error(new_output, target)
        error_derivative = (error - new_error)/(2*delta)
        self.weights[j][i]-=delta
        return error_derivative


    def symbolic_error_derivative(self, input, target, i, j): #not agreeing with numerical
        error_derivative = -1*input[i]*(target[j]-self.compute_output(input)[j])
        return error_derivative

    def learn_method_1(self, input, target):
        error = 100
        iter = 0
        while error >= 1:
            output = compute_output(input)
            self.weights
            for i in range(self.input_length):
                for j in range(self.output_length):
                    self.weights[j][i] -= self.numeric_derivative_error_right(input, target, i, j, compute_error(input, target))*self.weights[j][i]
                    debug(error)
                print(i)
            error = self.compute_error(self.compute_output(input), target)
            iter += 1

        return error
        return self.compute_output(input)
def debug(*x):
    for i in x:
        print(i)
def main():
    n = 50
    target = [0,0,1,0,0,0,0,0,0,0]
    a = neuralnet(n, len(target))

    input = [random.random() for i in range(n)] #784 can be replaced by len(input), 10 by output_length
    output = a.compute_output(input)
    error = a.compute_error(output, target)
    error_derivative_numeric = a.numeric_derivative_error(input, target, 0, 0)
    error_derivative_symbolic = a.symbolic_error_derivative(input, target, 0, 0)
    debug("error: ",error)
    debug("output 1: ", output)
    debug("numeric: ",error_derivative_numeric)
    debug("symbolic: ",error_derivative_symbolic)
    #error = a.learn_method_1(input, target)
    #debug("error 2:", error)

    #debug("output 2:", output)
main()
