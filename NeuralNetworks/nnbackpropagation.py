import sys
import time
import math
import random

def make_list(lsize):
    thelist = []
    for i in range(lsize):
        thelist.append(0)
    return thelist

def make_matrix(rows, columns):
    matrix = []
    for i in range(rows):
        matrix.append(make_list(columns))
    return matrix

def rand(a , b):
    ran = random.uniform(a, b)
    return ran

class NeuralNetwork:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1 # add a bias node
        self.nh = nh
        self.no = no

        # activation
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no
        
        # weight
        self.wi = make_matrix(self.ni, self.nh)
        self.wo = make_matrix(self.nh, self.no)
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-2.0, 2.0)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)
 
        self.ci = make_matrix(self.ni, self.nh)
        self.co = make_matrix(self.nh, self.no)

    def update(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError, 'wrong number of inputs'

        #input activation
        for i in range(self.ni-1):
            #self.ai[i] = 1.0/(1.0+math.exp(-inputs[i]))
            self.ai[i] = inputs[i]

        #hiddin activation
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = 1.0/(1.0+math.exp(-sum))

        #output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = 1.0/(1.0+math.exp(-sum))

        return self.ao[:]


    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError, 'wrong number of target values'

        #output error
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            ao = self.ao[k]
            output_deltas[k] = ao*(1-ao)*(targets[k]-ao)

        # hidden error
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            sum = 0.0
            for k in range(self.no):
                sum = sum + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = self.ah[j]*(1-self.ah[j])*sum

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change
                #print N*change, M*self.co[j][k]

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            delta = targets[k]-self.ao[k]
            error = error + 0.5*delta*delta
        return error


    def test(self, patterns):
        for p in patterns:
            if PRINT_IT:
                print p[0], '->', self.update(p[0])

    def weights(self):
        if PRINT_IT:
            print 'Input weights:'
            for i in range(self.ni):
                print self.wi[i]
            print
            print 'Output weights:'
            for j in range(self.nh):
                print self.wo[j]

    def train(self, patterns, iterations=2000, LR=0.5, MF=0.1):
        # LR: learning rate
        # MF: momentum factor
        for i in xrange(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, LR, MF)
            if PRINT_IT and i % 100 == 0:
                print 'error', error


def demo():
    # Teach network XOR
    pat = [
        [[0,0], [0]],
        [[0,1], [1]],
        [[1,0], [1]],
        [[1,1], [0]]
    ]

    # create a network
    n = NeuralNetwork(2, 3, 1)
    n.train(pat, 2000)
    n.test(pat)
