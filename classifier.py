print "\n\nWith the processed database, let's now run some logistic regression."
raw_input('enter to continue')

import lrdemo
lrdemo.demo()

print "\n\nThere's a chart of all the weights and what they mean, along with a list of every player that was classified in results.txt."
print "It always seems to positively weigh 3-pt % with centers, which is kind of weird. I guess they never take shots, but when they do, it's because they are wide open and feel like they can make it. Feel free to look at the stats in results.txt and try and explain why they are the way they are."
print "Anyway, these resutls indicate a few things to me. One is that logistic regression alone like this isn't good enough. I'm sure I could get 80-85% with a smarter system. But how?"
print "I tried writing a neural network, for fun, but there's a problem with it."
raw_input('enter to continue')

import netset
netset.demo()

print "Well, that's all. Thanks for using the bball player classifier!"
