from numpy import *
import random
import pdb
#from scipy.optimize import fmin_cg

# http://www.scipy.org/NumPy_for_Matlab_Users/

'''
In general I keep matrices in the form:
  <-- features -->
^
|
d
a
t
a
|
aka (data, features) whenever possible
'''

# sigmoid function
def sig(x):
    return 1.0 / (1.0 + exp(-x))

# sigmoid derivative
def dsig(x):
    if type(x) == matrix:
        return multiply((1 - sig(x)), sig(x))
    else:
        return (1 - sig(x)) * sig(x)
    
def rand(a, b):
    return (b-a)*random.random() + a

class NN:
    def __init__(self, in_n, hid_n, out_n, lam = 0.5, epsilon = 0.2):
        self.lam = float(lam)
        self.epsilon = float(epsilon)
        self.in_n = in_n
        self.hid_n = hid_n
        self.out_n = out_n
        
        # initialize everything to random weights
        vector = []
        for i in xrange((in_n+1) * hid_n):
            vector.append(rand(-epsilon, epsilon))
        self.hid_t = mat(vector).reshape(hid_n, in_n+1)
        
        vector = []
        for i in xrange((hid_n+1) * out_n):
            vector.append(rand(-epsilon, epsilon))
        self.out_t = mat(vector).reshape(out_n, hid_n+1)
        
    def train(self, x, y_vals, iters = 1000, rate = 0.05):
        for i in xrange(iters):
            self.forward_prop(x)
            print "Cost at iteration %r: %r" % (i, self.cost(x, y_vals, True))
            self.back_prop(y_vals)
            self.hid_t -= rate * self.hid_t_grad
            self.out_t -= rate * self.out_t_grad
            
        print "\n\tFinal cost: %r" % self.cost(x, y_vals)
        
    def cost(self, x, y_vals, skip = False):
        if not skip:
            h = self.forward_prop(x)
        else:
            h = self.hypothesis
        y = mat(y_vals)
        assert shape(h) == shape(y)
        
        m = float(shape(h)[0])
        
        reg_term = (self.lam/(2.0*m)) * (sum(self.hid_t[:,1:]) + sum(self.out_t[:,1:]))
        j = (1.0/m) * sum( multiply(log(h),(-y)) - multiply(log(1-h), (1-y)) ) + reg_term
        
        return j
    
    # this computes a numerical gradient and compares it to the gradient
    # values that back_prop is creating
    def test_gradient(self, x, y_vals, epsilon = 0.0001):
        self.forward_prop(x)
        self.back_prop(y_vals)
        hid_t_numgrad = self.hid_t * 0
        out_t_numgrad = self.out_t * 0
        
        for i in xrange(shape(self.hid_t)[0]):
            for j in xrange(shape(self.hid_t)[1]):
                self.hid_t[i,j] += epsilon
                J1 = self.cost(x, y_vals)
                self.hid_t[i,j] -= epsilon*2
                J2 = self.cost(x, y_vals)
                hid_t_numgrad[i,j] = (J1 - J2) / (2 * epsilon)
                self.hid_t[i,j] += epsilon
                
        for i in xrange(shape(self.out_t)[0]):
            for j in xrange(shape(self.out_t)[1]):
                self.out_t[i,j] += epsilon
                J1 = self.cost(x, y_vals)
                self.out_t[i,j] -= epsilon*2
                J2 = self.cost(x, y_vals)
                out_t_numgrad[i,j] = (J1 - J2) / (2 * epsilon)
                self.out_t[i,j] += epsilon
                
        n_roll = concatenate((reshape(hid_t_numgrad, -1), reshape(out_t_numgrad, -1)), 1)
        b_roll = concatenate((reshape(self.hid_t_grad, -1), reshape(self.out_t_grad, -1)), 1)
        numer = n_roll - b_roll
        denom = n_roll + b_roll
        diff = float(sqrt(numer * numer.transpose())) / float(sqrt(denom * denom.transpose()))
        
        print "The backpropogation algorithm is working correctly if the "
        print "following number is smaller than 0.00000001 or so: %r" % diff
        
    def train_numgrad(self, x, y_vals, iters = 500, rate = 0.5, epsilon = 0.00001):
        for i in xrange(iters):
            self.forward_prop(x)
            print "Cost at iteration %r: %r" % (i, self.cost(x, y_vals, True))
            hid_t_numgrad = self.hid_t * 0
            out_t_numgrad = self.out_t * 0
            for i in xrange(shape(self.hid_t)[0]):
                for j in xrange(shape(self.hid_t)[1]):
                    self.hid_t[i,j] += epsilon
                    J1 = self.cost(x, y_vals)
                    self.hid_t[i,j] -= epsilon*2
                    J2 = self.cost(x, y_vals)
                    hid_t_numgrad[i,j] = (J1 - J2) / (2 * epsilon)
                    self.hid_t[i,j] += epsilon
                    
            for i in xrange(shape(self.out_t)[0]):
                for j in xrange(shape(self.out_t)[1]):
                    self.out_t[i,j] += epsilon
                    J1 = self.cost(x, y_vals)
                    self.out_t[i,j] -= epsilon*2
                    J2 = self.cost(x, y_vals)
                    out_t_numgrad[i,j] = (J1 - J2) / (2 * epsilon)
                    self.out_t[i,j] += epsilon
            self.hid_t -= rate * hid_t_numgrad
            self.out_t -= rate * out_t_numgrad
        print "\n\tFinal cost: %r" % self.cost(x, y_vals)
    
    # based on the most recent forward-prop activation values
    # if forward-prop hasn't been run, there will be an error because
    # self.a_hid et al will not have been initialized
    def back_prop(self, y_vals):
        h = self.hypothesis
        y = mat(y_vals)
        assert shape(h) == shape(y)
        m = float(shape(h)[0])
        
        d_out = h - y
        #print shape(d_out)
        self.out_t_grad = (self.a_hid * d_out).transpose() / m
        #print shape(self.out_t_grad)
        self.out_t_grad[:,1:] += (self.lam / m) * self.out_t[:,1:]
        #self.out_t_grad = self.out_t_grad.reshape(-1) # unrolls matrix
        #print shape(d_out)
        #print shape(self.hid_t)
        #print shape(self.out_t)
        d_hid = (d_out * self.out_t).transpose()
        #print shape(d_hid)
        d_hid = multiply(d_hid[1:,:], dsig(self.z_hid))
        #print shape(d_hid)
        self.hid_t_grad = (d_hid * self.a_in) / m
        #print shape(self.hid_t_grad)
        self.hid_t_grad[:,1:] += (self.lam / m) * self.hid_t[:,1:]
        #self.hid_t_grad = self.hid_t_grad.reshape(-1) # unrolls matrix
        #return concatenate((self.hid_t_grad, self.out_t_grad), 1)
    
    def forward_prop(self, x):
        inputs = mat(x)
        assert shape(inputs)[1] == self.in_n
        m = shape(inputs)[0]
        self.a_in = concatenate((mat(ones(m)).transpose(), inputs), 1) # bias unit
        assert shape(self.a_in) == (m, self.in_n+1)
        self.z_hid = self.hid_t * self.a_in.transpose()
        assert shape(self.z_hid) == (self.hid_n, m)
        o = mat(ones((1, m)))
        self.a_hid = concatenate((o, sig(self.z_hid)), 0)
        assert shape(self.a_hid) == (self.hid_n+1, m)
        z_out = self.out_t * self.a_hid
        self.hypothesis = sig(z_out).transpose()
        assert shape(self.hypothesis) == (m, self.out_n)
        return self.hypothesis
        
    def predictions(self):
        return self.hypothesis.max(1) == self.hypothesis
        
