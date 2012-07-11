from numpy import *
from logreg import *

def demo():
    with open('trainingset') as data:
        t = data.read()
        t_set = t.strip().split('\n')
        del(t)
        y = []
        x = []
        names = []
        i = 0
        for s in t_set:
            if s[0] == 'C':
                y.append([1,0,0])
            elif s[0] == 'F':
                y.append([0,1,0])
            else:
                y.append([0,0,1])
            s = s.split('\t')
            names.append(s[1])
            try:
                x.append([float(f) for f in s[2:]])
            except:
                print "Error at data point: \n" + s[2:]
            i += 1
            
    with open('testset') as data:
        t = data.read()
        t_set = t.strip().split('\n')
        del(t)
        y_t = []
        x_t = []
        names = []
        i = 0
        for s in t_set:
            if s[0] == 'C':
                y_t.append([1,0,0])
            elif s[0] == 'F':
                y_t.append([0,1,0])
            else:
                y_t.append([0,0,1])
            s = s.split('\t')
            names.append(s[1])
            try:
                x_t.append([float(f) for f in s[2:]])
            except:
                print "Error at data point: \n" + s[2:]
            i += 1
    #lam = [0.01, 0.03, 0.3, 1, 3]
    #for l in lam:
    lr = LR(16, 3, 0.05)
    lr.train(x, y, 1000, False)
    print "Data has been imported, and the classifier has been trained. Let's test the results on the test set:"
    raw_input('enter to continue')

    predictions = lr.predictions(x_t)

    with open('results.txt', 'w') as doc:
        weights = lr.theta
        cs = weights[0,:].tolist()[0]
        fs = weights[1,:].tolist()[0]
        gs = weights[2,:].tolist()[0]
        doc.write("Here are the weights of individual stats: \n")
        doc.write("\t bias\theig\tweig\tGP  \tMPG \tFG% \t3pt% \tFT% \tOFF \tDEF \tRPG \tAPG \tSPG \tBPG \tTO  \tPF  \tPPG\n")
        doc.write("C:\t{0:.2f}\t{1:.2f}\t{2:.2f}\t{3:.2f}\t{4:.2f}\t{5:.2f}\t{6:.2f}\t{7:.2f}\t{8:.2f}\t{9:.2f}\t{10:.2f}\t{11:.2f}\t{12:.2f}\t{13:.2f}\t{14:.2f}\t{15:.2f}\t{16:.2f}\n".format(*cs))
        doc.write("F:\t{0:.2f}\t{1:.2f}\t{2:.2f}\t{3:.2f}\t{4:.2f}\t{5:.2f}\t{6:.2f}\t{7:.2f}\t{8:.2f}\t{9:.2f}\t{10:.2f}\t{11:.2f}\t{12:.2f}\t{13:.2f}\t{14:.2f}\t{15:.2f}\t{16:.2f}\n".format(*fs))
        doc.write("G:\t{0:.2f}\t{1:.2f}\t{2:.2f}\t{3:.2f}\t{4:.2f}\t{5:.2f}\t{6:.2f}\t{7:.2f}\t{8:.2f}\t{9:.2f}\t{10:.2f}\t{11:.2f}\t{12:.2f}\t{13:.2f}\t{14:.2f}\t{15:.2f}\t{16:.2f}\n".format(*gs))
            
        doc.write("Name of player \t\t  real|predicted\n")
        g_errors = 0
        f_errors = 0
        c_errors = 0
        error = False
        for i in xrange(len(names)):
            s = names[i]
            s += (' ' * (24 - len(s)))
            r = y_t[i]
            p = predictions[i].tolist()[0]
            if r[0] == 1:
                r = 'C'
            elif r[1] == 1:
                r = 'F'
            else:
                r = 'G'
            if p[0] == 1:
                p = 'C'
                if p != r:
                    c_errors += 1
                    error = True
            elif p[1] == 1:
                p = 'F'
                if p != r:
                    f_errors += 1
                    error = True
            else:
                p = 'G'
                if p != r:
                    g_errors += 1
                    error = True
            doc.write(s + r + '\t' + p + '\n')

    errors = f_errors + c_errors + g_errors
    total = len(names)
    right = total - errors
    acc = right/float(total)

    print "\nAccuracy: %r (%r out of %r)" % (acc, right, total)
    print "Percentage of errors on centers: %r" % (c_errors/float(errors))
    print "Percentage of errors on guards: %r" % (g_errors/float(errors))
    print "Percentage of errors on forwards: %r" % (f_errors/float(errors))
        #print "At lambda %r.\n" % l

