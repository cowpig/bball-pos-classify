from numpy import *
from neuralnet import *
#import scipy
#from nn import *

def demo():
    with open('trainingset') as data:
        t = data.read()
        t_set = t.strip().split('\n')
        del(t)
        y = []
        x = []
        i = 0
        for s in t_set:
            if s[0] == 'C':
                y.append([1,0,0])
            elif s[0] == 'F':
                y.append([0,1,0])
            else:
                y.append([0,0,1])
            try:
                x.append([float(f) for f in s.split('\t')[2:]])
            except:
                print "Error at data point: \n" + s.split('\t')[2:]
            i += 1
            
    net = NN(16, 25, 3, 0.1)

    net.train(x, y, 400)
    acc = float(sum(net.predictions() == y)) / float(shape(y)[0] * shape(y)[1])
    print "\nAccuracy: %r\n" % acc
    print "As you can see, the network doesn't converge. This is because there's a bug in the backpropogation that I couldn't find, despite hours and hours of trying to find it... Ugh. Anyway, here's the proof:"
    raw_input('enter to continue')
    net = NN(16, 25, 3, 0.1)
    net.test_gradient(x, y)
