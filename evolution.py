from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

import numpy as np
import pandas as pd
import random

from KerasGA import GeneticAlgorithm

population_size = 10
generations = 5

model = keras.Sequential([
    layers.Dense(6,input_dim=2, activation='relu', name='fc1'),
    layers.Dense(1,activation='sigmoid', name='output')])

model.compile(optimizer=keras.optimizers.Adam(0.01),loss='categorical_crossentropy', metrics=['accuracy'])

GA = GeneticAlgorithm(model,population_size= population_size,selection_rate = 0.1, mutation_rate = 0.2)

def init():
    population = GA.initial_population()

    np.save('./data/population.npy',population)
    np.save('./data/scores.npy',[])
    # model.save('./models/agent.h5')

    print("Initialisation done!")

def load_weights(chromosome):
    # model = load_model('./models/agent.h5')
    chromosomes = np.load('./data/population.npy',allow_pickle=True)

    model.set_weights(chromosomes[chromosome])

    # model.save('./models/agent.h5')
    print("weights loaded for id : {}".format(chromosome))
    return model

def evolution():
    population = np.load('./data/population.npy',allow_pickle=True)
    scores = np.load('./data/scores.npy',allow_pickle=True).tolist()

    # Selection
    top_performers = GA.strongest_parents(population,scores)

    # Make pairs
    pairs = []
    while len(pairs) != GA.population_size:
        pairs.append(GA.pair(top_performers))

    # Crossover
    base_offsprings = []
    for pair in pairs:
        offsprings = GA.crossover(pair[0][0], pair[1][0])
        base_offsprings.append(offsprings[-1])

    # Mutation
    new_population = GA.mutation(base_offsprings)

    np.save('./data/population.npy',new_population)
    np.save('./data/scores.npy',[])