Basketball Player Classifier
-----
System for predicting basketball players' positions based on their statistics.

Installing Requirements
========
Unpack everything into some folder, and then
```shell
$ pip install -r requirements.txt
```
It'll install the latest versions of Numpy, Scipy, and BeautifulSoup, all of which are required.

Obtaining Data
========
Run 
```shell
$ python get_data.py
```
to obtain basketball player data off of yahoo! sports and NBA.com and prepare it for use in the classifier.

Classifying Data
========
Run
```shell
$ python classify.py
```
to run a simple logistic regression classifier on the data, and then a neural network classifier. The NN has a bug in the backprop that prevents it from optimizing correctly.
