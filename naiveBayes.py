# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import math as m

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **

  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """

    # might be useful in your code later...
    # this is a list of all features in the training set.
    self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));

    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]

    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)

  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):

    #We'll use this for P(0), P(1)...P(9) or face
    prior = util.Counter()
    for x in trainingLabels:
      prior[x] = prior[x]+ 1
    prior.normalize()
    self.prior = prior
    individualcount = {}
    total = {}

    for x in self.features:
      #go through each feature, how many times 0/1/2/3/4 appears
      individualcount[x] = {0: util.Counter(), 1: util.Counter(), 2: util.Counter()}
      total[x] = util.Counter() #useful for smoothing later on

    # Calculate totals and counts
    for x, datum in enumerate(trainingData):
      label = trainingLabels[x]
      for y, value in datum.items():
        individualcount[y][value][label] = individualcount[y][value][label] + 1.0
        total[y][label] = total[y][label] + 1.0

    bestAccuracy = None
    # Evaluate each k, and use the one that yields the best accuracy
    k = .001
    correct = 0
    smoothprobability = {}
    # for all features create 0-3 util counters
    # <feature>: {0: {}, 1: {}, 2: {}, 3: {}, 4: {}}
    for x in self.features:
      smoothprobability[x] = {0: util.Counter(), 1: util.Counter(), 2: util.Counter()}
    # smoothing
    # for all features
    for x in self.features:
      # for all 0-4 values of features
      for value in [0, 1, 2]:
        # for all 1-9
        for y in self.legalLabels:
          # smooth it out
          # basically holds the probability that a given lighted up feature is a certain label
          # P(phi[](x) | y = 1...9) => ex P(phi[0] | y(1) ) = Feature 1, (7,3) has a .96% chance to light my way 0 times for 0, 91%  for 4
          smoothprobability[x][value][y] = (individualcount[x][value][y] + k) / (total[x][y] + k*2)
          # Check the accuracy associated with this k
    self.smoothprobability = smoothprobability

  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())#from LogJoint you are given the log probability of 0-9, pick the lowest digit
      self.posteriors.append(posterior)

    return guesses

  def classify2(self, testData, validationLabels):
    """
    Classify the data based on the posterior distribution over labels.
    You shouldn't modify this method.
    """
    guesses = []
    a = 0
    self.argmax = [] # Log posteriors are stored for later data analysis (autograder).
    #self.
    for datum in testData:
      argmax = self.calculate(datum)
      guesses.append(argmax.argMax())#from LogJoint you are given the log probability of 0-9, pick the lowest digit
      self.argmax.append(argmax)
      for i, guess in enumerate(guesses):
        if validationLabels[i] != guess:
          a += 1
    return guesses

  def calculateLogJointProbabilities(self, datum):
    logJoint = util.Counter()

    "*** YOUR CODE HERE ***"
    #for all labels
    for x in self.legalLabels:
      logJoint[x] = m.log(self.prior[x])
      for y in self.smoothprobability:
        prob = self.smoothprobability[y][datum[y]][x]
        if prob > 0:
          logJoint[x] += m.log(prob)
    return logJoint

  def calculate(self, datum):

    logJoint = util.Counter()

    "*** YOUR CODE HERE ***"
    # for all labels
    for x in self.legalLabels:
      logJoint[x] = m.log(self.prior[x])
      for y in self.naiveConditional:
        prob = self.naiveConditional[y][datum[y]][x]
        logJoint[x] += (prob and m.log(prob))
    return logJoint

  def findHighOddsFeatures(self, label1, label2):
    """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2)
    Note: you may find 'self.features' a useful way to loop through all possible features
    """
    featuresOdds = []

    for f in self.features:
      top = self.conditionals[f][1][label1]
      bottom = self.conditionals[f][1][label2]
      ratio = top / bottom
      featuresOdds.append((f, ratio))

    featuresOdds.sort()
    featuresOdds = [feat for val, feat in featuresOdds[-100:]]

    return featuresOdds