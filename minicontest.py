# minicontest.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import random
import math

class contestClassifier(classificationMethod.ClassificationMethod):
  """
  Create any sort of classifier you want. You might copy over one of your
  existing classifiers and improve it.
  """

  def __init__(self, legalLabels, max_iterations):
    self.legalLabels = legalLabels
    self.type = "minicontest"
    self.max_iterations = max_iterations
    self.hiddenWeights = {} # 784 * 10 of them
    self.hiddenLayerNeurons = [1,2,3,4,5,6,7,8,9,10] # could only come up with 10 distinct numeric features
    self.hiddenBias = {}
    self.outerBias = {}
    self.outerWeights = {} # 10 from legalLabels * 10 from hiddenNeurons
    for label in legalLabels:
      self.hiddenWeights[label] = util.Counter()  # this is the data-structure you should use
    for label in self.hiddenLayerNeurons:
      self.outerWeights[label] = util.Counter()

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    The training loop for the "neural network" passes through the training data several
    times and updates the weight vector for each hiddenLayerLabel and then legalLabel based on classification errors.
    """
    # obtain datum coordinates
    self.features = trainingData[0].keys()

    # initialize random weights and biases
    for label in self.hiddenLayerNeurons:
      self.hiddenBias[label] = random.randrange(4)
      for weight in self.features:
        self.hiddenWeights[label][weight] = random.randrange(100)/100.0
    for label in self.legalLabels:
      self.outerBias[label] = random.randrange(4)
      for weight in self.hiddenLayerNeurons:
        self.outerWeights[label][weight] = random.randrange(100)/100.0

    for percentOfData in range (10):
      numDatums = math.floor((percentOfData+1)*.1*len(trainingData))

      # for 
      for iteration in range (1):
        numIncorrect = 0
        # parse through a percentage of the data
        for i in range(numDatums):

    # 1: 0
    # 2: dash on top of 1 and middle of 7
    # 3:



  def classify(self, testData):
    """
    Please describe how data is classified here.
    """
    guesses = []
    for datum in testData:
      vectors = util.Counter()
      for l in self.legalLabels:
          vectors[l] = self.outerWeightsweights[l] * ??? + self.bias[l]
      guesses.append(vectors.argMax())
    return guesses