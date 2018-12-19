#!/usr/bin/env python
# coding: utf-8

# In[6]:


from subprocess import check_output
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.model_selection import  train_test_split
import time #helper libraries
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from numpy import newaxis
import numpy as np
import pandas as pd
import math
import talib
import datetime
import sys
from multiprocessing import Pool
import multiprocessing
np.set_printoptions(precision = 2)
cores = multiprocessing.cpu_count()

logFile = "log " + datetime.datetime.now().strftime("%I:%M%p %B %d, %Y") + ".txt"
with open(logFile, "w+") as f:
    f.write("")

def log(msg):
    with open(logFile, 'a') as f:
        f.write(msg + '\n')
    print(msg)

print("Cores: " + str(cores))
print("Initialized")


# In[ ]:


class Genome:
    #params define the range of allowed values for hyperparameters
    #if params for a hyperparameter are [a, b, c], the hyperparameter mutates as c*(np.random.randint(a)+b)
    #params = [[max, offset, scale factor]]
    params = {}
    params["number_per_model"] = [255, 1, 1]
    params["LOOK_BACK"] = [50, 1, 1]
    params["VALIDATION_SPLIT"] = [30, 0, 0.01]
    params["RSI_window"] = [60, 3, 1]
    params["PREDICT_DAYS_AHEAD"] = [255, 1, 1]
    params["LAYER_1"] = [20, 5, 10]
    params["LAYER_2"] = [20, 5, 10]
    
    def __init__(self, genes_dict):
        if genes_dict == 'random':
            self.random()
        else:
            self.genes_dict = genes_dict
        return
    
    def random(self):
        self.genes_dict = {}
        for param in Genome.params.keys():
            a = Genome.params[param][0]
            b = Genome.params[param][1]
            c = Genome.params[param][2]
            self.genes_dict[param] = (np.random.randint(a) + b)*c
        return
    
    def __getitem__(self, key):
        return self.genes_dict[key]
    
    def mutate(self):
        for key in self.genes_dict.keys():
            if np.random.randint(len(self.genes_dict.keys())) == 0:
                a = Genome.params[key][0]
                b = Genome.params[key][1]
                c = Genome.params[key][2]
                self.genes_dict[key] = c*(np.random.randint(a)+b)
        return

def cross_over(parents):
    parent_a, parent_b = parents
    offspring = {}
    for key in parent_a.genes.genes_dict.keys():
        a = np.random.randint(2)
        if a == 1:
            offspring[key] = parent_a.genes[key]
        elif a == 0:
            offspring[key] = parent_b.genes[key]
        else:
            raise("Invalid")
    offspring = Genome(offspring)
    offspring.mutate()
    return Individual(offspring)
        
        

class Individual:
    def __init__(self, x):
        if x == 'random':
            self.genes = Genome('random')
        else:
            self.genes = x
        
        self.number_per_model = self.genes["number_per_model"]
        self.LOOK_BACK = self.genes["LOOK_BACK"]
        self.VALIDATION_SPLIT = self.genes["VALIDATION_SPLIT"]
        self.RSI_window = self.genes["RSI_window"]
        self.PREDICT_DAYS_AHEAD = self.genes["PREDICT_DAYS_AHEAD"]
        self.LAYER_1 = self.genes["LAYER_1"]
        self.LAYER_2 = self.genes["LAYER_2"]
        return
    
    def printGenes(self):
        log(self.genes.genes_dict)
        return
    def train(self):
        accuracy = []
        for day in range(0, 500, self.number_per_model):
            prices = pd.read_csv('./prices-split-adjusted.csv') #get data
            stock = prices.loc[prices['symbol'] == 'AAPL'].copy() #get only GOOG prices
            stock["Daily Returns"] = (stock.close - stock.close.shift(1))/stock.close.shift(1)
            stock["y"] = (stock.close - stock.close.shift(self.PREDICT_DAYS_AHEAD))/stock.close.shift(self.PREDICT_DAYS_AHEAD) #create returns (called y since that's what we're trying to predict)
            #Add features
            for i in range(1, self.LOOK_BACK+1):
                #Add column for returns of n days ago (e.g. yesterday's returns, the returns of the day before yesterday...)
                stock[str(i) + " Days Ago"] = stock["Daily Returns"].shift(i)
            stock["RSI"] = talib.RSI(stock.close, self.RSI_window)
            technical_indicators = ["RSI"]
            stock = stock[['y'] + [str(i) + " Days Ago" for i in range(1, self.LOOK_BACK + 1)] + technical_indicators].copy().dropna() #drop the other columns

            #Scale and separate test and train data
            train_length = 1000
            train_start = day
            train_stop = train_start+train_length
            train = stock.loc[stock.index[0:train_stop]]
            zero = 'a'
            for col in stock.columns: 
                scaler = MinMaxScaler(feature_range=(0.5, 0.95))
                scaler.fit(train[[col]]) #!!!!!!!!!!!!!!!! ONLY FIT SCALER ON TRAINING DATA, FITTING ON TEST DATA AS WELL IS CHEATING !!!!!!!!!!!!
                stock[col] = scaler.transform(stock[[col]]) #apply scaler to all data 
                if col == 'y':
                    zero = scaler.transform(np.reshape(np.array(0), (1, 1)))[0][0]
            #Actually separate test data
            stock = np.array(stock.values)
            x_train = stock[train_start:train_stop, 1:]
            test_stop = train_stop + self.number_per_model
            x_test = np.array(stock[train_stop:test_stop, 1:])
            y_train = stock[train_start:train_stop, 0]
            y_test = np.array(stock[train_stop:test_stop, 0])

            x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
            x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
            

            #Define model architecture (copied from someone else's model, will optimize with a genetic algorithm later)
            model = Sequential()
            model.add(LSTM(input_shape=(1, x_train.shape[2]), units = self.LAYER_1, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(self.LAYER_2, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(units=1))
            model.add(Activation('linear'))

            #Compile, fit on training data, and evaluate on test data
            model.compile(loss='mse', optimizer='rmsprop')
            model.fit(x_train, y_train, batch_size = 128, epochs=10, validation_split = self.VALIDATION_SPLIT, verbose = 0)
            #model.evaluate(x_test, y_test, verbose = 0)
            y_predict = model.predict(x_test)
            for i in range(len(y_test)):
                correct = int(not(bool(y_predict[i][0] > zero) ^ bool(y_test[i] > zero)))
                accuracy.append(correct)
            self.fitness = sum(accuracy)/len(accuracy)
        self.printRules()
        log("Fitness: " + str(self.fitness))
        log("")
        return self
    
    def printRules(self):
        log("Days In Between Training: " + str(self.number_per_model))
        log("Previous Days of Returns Included as Features: " + str(self.LOOK_BACK))
        log("Validation Split: " + str(self.VALIDATION_SPLIT))
        log("RSI Length: " + str(self.RSI_window))
        log("Days ahead to predict: " + str(self.PREDICT_DAYS_AHEAD))
        log("LSTM Layer 1 Size: " + str(self.LAYER_1))
        log("LSTM Layer 2 Size: " + str(self.LAYER_2))
        return
    
class GeneticAlgorithm:
    PROPORTION_TO_KILL = 0.5
    CARRYING_CAPACITY = 24
    CHILDREN_PER_COUPLE = 2
    STARTING_POPULATION = 12
    def __init__(self):
        #create population
        self.population = []
        offspring_list = []
        for i in range(GeneticAlgorithm.STARTING_POPULATION):
            model = Individual('random')
            offspring_list.append(model)

        p = Pool(cores)
        
        log("Created Initial Population")
        self.population += p.map(Individual.train, offspring_list)
        return
    
    def kill(self):
        log("Killing least fit members...")
        self.population.sort(key = lambda x: x.fitness)
        cutoff = math.floor(len(self.population) * GeneticAlgorithm.PROPORTION_TO_KILL)
        self.population = self.population[cutoff:]
    
    def chooseParents(self):
        #random
        return self.population[np.random.randint(len(self.population))], self.population[np.random.randint(len(self.population))]
    
    def mate(self):
        log("Mating remaining members...")
        size_before_mating = len(self.population)
        pop = len(self.population)
        offspring_list = []
        while(pop < GeneticAlgorithm.CARRYING_CAPACITY and pop < (size_before_mating/GeneticAlgorithm.PROPORTION_TO_KILL + 2)):
            parent_a, parent_b = self.chooseParents()
            for i in range(GeneticAlgorithm.CHILDREN_PER_COUPLE):
                offspring = cross_over((parent_a, parent_b))
                offspring_list.append(offspring)
                pop += 1
        
        #train offspring in parallel
        p = Pool(cores)
        self.population += p.map(Individual.train, offspring_list)
                
    def getPopulationFitness(self):
        return np.mean([member.fitness for member in self.population])
    
    def trainGenerations(self, generations):
        for gen in range(generations):
            log("")
            log("Generation: " + str(gen))  
            log("Population Fitness: " + str(self.getPopulationFitness()))
            log("Population Size: " + str(len(self.population)))
            log("")
            self.kill()
            self.mate()
algo = GeneticAlgorithm()
algo.trainGenerations(10)

