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
        self.features = trainingData[0].keys()  # could be useful later
        trueWeight = {}
        bias = {}
        perceptronScore = 0

        for y in self.legalLabels:
          trueWeight[y] = util.Counter()
          bias[y] = 0
          for value in self.features:
            trueWeight[y][value] = random.randrange(0,2)
        self.weights = trueWeight
        self.bias = bias
        maxrange = len(trainingData)
        for iteration in range(self.max_iterations):
            incorrect = 0
            print "Starting iteration ", iteration, "..."
            randomrange = list(range(maxrange))
            random.shuffle(randomrange)
            for i in range(len(trainingData)):
                "*** YOUR CODE HERE ***"
                highestscore = None
                myY = None
                datum = trainingData[randomrange[i]]
                #datum = trainingData[i]
                scorearray = {}
                for y in self.legalLabels:
                    score = datum * self.weights[y] + bias[y]
                    if score > highestscore or highestscore is None:
                        highestscore = score
                        myY = y
                    perceptronScore = datum * trueWeight[y] + bias[y]
                    scorearray[y] = perceptronScore
                actualY = trainingLabels[randomrange[i]]
                #actualY = trainingLabels[i]
                # Wrong guess, update weights
                # actualY is the training label
                # myY is the y that i get  from list
                while myY != actualY:
                    bias[actualY] = bias[actualY] + 1
                    self.bias[actualY] = bias[actualY]
                    self.weights[actualY] = self.weights[actualY] + datum

                    bias[myY] = bias[myY] - 1
                    self.bias[myY] = bias[myY]
                    self.weights[myY] = self.weights[myY] - datum

                    score = (datum * self.weights[myY]) + bias[myY]
                    scorearray[myY] = score
                    score = (datum * self.weights[actualY]) + bias[actualY]
                    scorearray[actualY] = score
                    myY = max(scorearray, key=scorearray.get)
                    incorrect += 1
            print(incorrect)
            if incorrect == 0:
                break


    def findHighWeightFeatures(self, label):
        """
    Returns a list of the 100 features with the greatest weight for some label
    """
        featuresWeights = []

        "*** YOUR CODE HERE ***"

        return featuresWeights