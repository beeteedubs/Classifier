# perceptron.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# Perceptron implementation
import util
import random
import numpy as np
import statistics
import time

PRINT = True


class PerceptronClassifier:
    """
  Perceptron classifier.
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """

    def __init__(self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.weights = {}
        for label in legalLabels:
            self.weights[label] = util.Counter()  # this is the data-structure you should use

    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels);
        self.weights = weights;

    def classify(self, data):
        """
    Classifies each datum as the label that most closely matches the prototype vector
    for that label.  See the project description for details.
    Recall that a datum is a util.counter...
    """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum + self.bias[l]
            guesses.append(vectors.argMax())
        return guesses

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
    The training loop for the perceptron passes through the training data several
    times and updates the weight vector for each label based on classification errors.
    See the project description for details.
    Use the provided self.weights[label] data structure so that
    the classify method works correctly. Also, recall that a
    datum is a counter from features to values for those features
    (and thus represents a vector a values).
    """

        self.features = trainingData[0].keys()  # could be useful later
        # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
        # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.
        trueWeight = {}  # dictionary of keys = label, value = dictionary of key = feature and value = weight
        bias = {}  # dictionary of key representing label, value represents numeric bias

        # creates randomized weights and biases for corresponding labels
        for y in self.legalLabels:
            trueWeight[y] = util.Counter()
            bias[y] = random.randrange(4)
            for value in self.features:
                trueWeight[y][value] = random.random()
        self.weights = trueWeight
        self.bias = bias

        for percentageOfData in range(10):  # run 10 times [0,9], each time accumulating 10% more data
            print "\n\n"
            print "Using ", (percentageOfData + 1) * 10, "% of training data"
            print "\n\n"
            for iteration in range(self.max_iterations):  # cap at 10, feels kinda safe
                numIncorrect = 0  # counts number of incorrect guesses
                numDatums = ((percentageOfData + 1) * 500)  # (10 datums in 1st iteration, 20 in 2nd etc...)
                print "Starting iteration ", (iteration + 1), "..."
                for i in range(numDatums):  # loop first "numDatum" datums
                    bestScore = None  # int corresponding score of machine's best guess
                    bestY = None  # represents machines guess
                    datum = trainingData[i]  # extract datum
                    scorearray = {}  # key = label, value = score
                    actualY = trainingLabels[i]  # correct label

                    for y in self.legalLabels:  # with the datum, have machine guess the number
                        b =  datum * self.weights[y]
                        score = datum * self.weights[y] + bias[y]
                        if score > bestScore or bestScore is None:
                            bestScore = score
                            bestY = y
                        scorearray[y] = score
                    while bestY != actualY:  # Wrong guess, update weights
                        # task 1: increase bias and weights for correct trueWeight label that were too low
                        # task 2: decrease machine's best guess's bias and weights

                        # task 1
                        bias[actualY] = bias[actualY] + 1
                        self.bias[actualY] = bias[actualY]
                        self.weights[actualY] = self.weights[actualY] + datum

                        # task 2
                        bias[bestY] = bias[bestY] - 1
                        self.bias[bestY] = bias[bestY]
                        self.weights[bestY] = self.weights[bestY] - datum

                        score = (datum * self.weights[bestY]) + bias[bestY]
                        scorearray[bestY] = score
                        score = (datum * self.weights[actualY]) + bias[actualY]
                        scorearray[actualY] = score
                        bestY = max(scorearray, key=scorearray.get)
                        numIncorrect += 1

                print "number of incorrect on iteration: ", numIncorrect
                if numIncorrect == 0 or numIncorrect/numDatums < 0.1:  # correct, so don't need to run so many times
                   break

"""
            ##########MORE TESTING#############
            ##EPIPHANY##
            ##EPIPHANY##
            ##EPIPHANY##
            numIncorrect = 0
            dictionaryForLabels = []

            for y in self.legalLabels:  # reorders training labels by least to greatest
                line = []
                for i in range(len(trainingLabels)):
                    if trainingLabels[i] == y:
                        line.append(i)
                dictionaryForLabels.append(line)

            i = 0
            j = 0
            print("stuck")
            while j < len(dictionaryForLabels): # same length as number of training
                epiphanylist = {}
                for y in self.features:
                    epiphanylist[y] = 0 # resets weights to 0???
                while i < len(dictionaryForLabels[j]):
                    datum = trainingData[dictionaryForLabels[j][i]]
                    for k, v in self.weights[0].items():
                        if datum[k] == 0:
                            epiphanylist[k] += 1
                    i += 1
                for k, v in self.weights[j].items():
                    if epiphanylist[k] == len(dictionaryForLabels[j]):
                        self.weights[j][k] = 0
                j += 1
            ########BULK
            ########BULK
            ########BULK
            x = 0
            j = 0
            i = 0
            while j < len(dictionaryForLabels):
                while i < len(dictionaryForLabels[j]):
                    datum = trainingData[dictionaryForLabels[j][i]]
                    scorearray = {}
                    for y in self.legalLabels:
                        perceptronScore = datum * trueWeight[y] + bias[y]
                        scorearray[y] = perceptronScore
                    actualY = trainingLabels[dictionaryForLabels[j][i]]
                    bestY = max(scorearray, key=scorearray.get)

                    # Wrong guess, update weights
                    while bestY != actualY:
                        self.bias[actualY] = self.bias[actualY] + 1
                        self.bias[actualY] = bias[actualY]
                        self.weights[actualY] = self.weights[actualY] + datum

                        self.bias[bestY] = self.bias[bestY] - 1
                        self.weights[bestY] = self.weights[bestY] - datum

                        score = (datum * self.weights[bestY]) + bias[bestY]
                        scorearray[bestY] = score
                        score = (datum * self.weights[actualY]) + bias[actualY]
                        scorearray[actualY] = score
                        bestY = max(scorearray, key=scorearray.get)
                        numIncorrect += 1
                        print("changed")
                    y = 0

                    datumcounter = 0
                    for k, v in self.weights[y].items():
                        if datum[k] > 0:
                            datumcounter = datumcounter + datum[k]

                    timeout = time.time() + 60 * 2

                    while y < 10:
                        # Basically: self.weights[y] = self.weights[y] - datum(.5)
                        # if other weights are too high
                        standardDeviation = statistics.stdev(scorearray[k] for k in scorearray)
                        standardAverage = statistics.mean(scorearray[k] for k in scorearray)
                        if scorearray.get(y) > (max(scorearray, key=scorearray.get) * .70) and y != actualY:
                            for k, v in self.weights[y].items():
                                self.weights[y][k] = v - datum[k] * .2
                            score = (datum * self.weights[y]) + bias[y]
                            scorearray[y] = score
                            numIncorrect += 1
                            print("decreasedy<s+s", actualY)
                            standardDeviation = statistics.stdev(scorearray[k] for k in scorearray)
                            standardAverage = statistics.mean(scorearray[k] for k in scorearray)
                            y = -1

                        # Basically: self.weights[actualY] = self.weights[actualY] * datum(.5)
                        # if max weight isz too high
                        if scorearray.get(actualY) < datumcounter * .9:
                            # Increase weights by 20%
                            for k, v in self.weights[actualY].items():
                                self.weights[actualY][k] = v + datum[k] * .5
                            score = (datum * self.weights[actualY]) + bias[actualY]
                            scorearray[actualY] = score
                            numIncorrect += 1
                            print("increased<", actualY)
                            y = -1

                        score = scorearray.get(max(scorearray, key=scorearray.get))

                        y += 1
                    datumcounter = 0
                    i += 1
                i = 0
                if numIncorrect != 0:
                    numIncorrect = 0
                    j = 0
                    i = 0
                else:
                    j += 1
            ###Last#######
            for iteration in range(self.max_iterations):
                numIncorrect = 0
                print "Starting iteration ", iteration, "..."
                randomrange = list(maxrange)
                random.shuffle(randomrange)
                for i in range(len(trainingData)):
                    bestScore = None
                    bestY = None
                    datum = trainingData[randomrange[i]]
                    scorearray = {}
                    for y in self.legalLabels:
                        score = datum * self.weights[y]
                        if score > bestScore or bestScore is None:
                            bestScore = score
                            bestY = y
                        # my way
                        perceptronScore = datum * trueWeight[y] + bias[y]
                        scorearray[y] = perceptronScore
                        # z = z + (dictionary.get((a,b)) * trueWeight[y].get((a,b)))
                    actualY = trainingLabels[randomrange[i]]
                    # Wrong guess, update weights
                    while bestY != actualY:
                        bias[actualY] = bias[actualY] + 1
                        self.bias[actualY] = bias[actualY]
                        self.weights[actualY] = self.weights[actualY] + datum

                        bias[bestY] = bias[bestY] - 1
                        self.bias[bestY] = bias[bestY]
                        self.weights[bestY] = self.weights[bestY] - datum

                        score = (datum * self.weights[bestY]) + bias[bestY]
                        scorearray[bestY] = score
                        score = (datum * self.weights[actualY]) + bias[actualY]
                        scorearray[actualY] = score
                        bestY = max(scorearray, key=scorearray.get)
                        numIncorrect += 1
                print(numIncorrect)
                if numIncorrect == 0:
                    break

            print("donzo")

            guesses = self.classify(validationData)
            correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
            print(correct)
"""
def findHighWeightFeatures(self, label):
        """
    Returns a list of the 100 features with the greatest weight for some label
    """
        featuresWeights = []

        "*** YOUR CODE HERE ***"
        # featuresWeights = self.weights[label].sortedKeys()[0z:100]
        featuresWeights = [k for k, v in sorted(self.weights[label].items(), key=lambda (k, v): (-v, k))][0:120]

        return featuresWeights
