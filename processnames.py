from random import shuffle

'''
separating the labeled data into training and test sets
input should come in the following format:

Player Name\tPosition\n

'''
def process(filename):
    with open(filename) as f:
        names = f.read()
        names = names.strip().split('\n')
        shuffle(names)
        
    assert len(names) > 100, "You should get more data."

    with open('trainingset', 'w') as train:
        with open('testset', 'w') as test:
            i = 0;
            for n in names:
                i += 1
                if i < int(len(names) * 0.7):
                    train.write(n + '\n')
                else:
                    test.write(n + '\n')
