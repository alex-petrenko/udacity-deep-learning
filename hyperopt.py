"""
Testing methods for hyperparameter optimization.

"""

import os
import sys
import math
import random
import shutil
import logging
import collections

from os.path import join

import numpy as np

from deap import base, creator, tools, algorithms

from utils import *
from dnn_utils import *
from dataset_utils import *


logger = logging.getLogger(os.path.basename(__file__))  # pylint: disable=invalid-name


creator.create('AccuracyMax', base.Fitness, weights=(1.0,))
creator.create('Individual', list, fitness=creator.AccuracyMax)


class EvolutionaryHyperopt:
    """Genetic algorithm for hyperparameter optimization."""

    def __init__(self):
        """Ctor."""
        self.param_grid = None
        self.initial_guess = None
        self.evaluation_func = None

        self.cache_hits = 0
        self.cache_misses = 0

        self.checkpoint_dir = None
        self.checkpoint_fname = None

        # evolution states
        self.generation = 0
        self.population = None
        self.cache = {}
        self.hof = None
        self.logbook = None

        self.initialized = False

    def set_param_grid(self, param_grid):
        """Initialize search space for individuals."""
        sorted_grid = sorted(param_grid.items(), key=lambda p: p[0])
        self.param_grid = collections.OrderedDict(sorted_grid)

    def set_initial_guess(self, initial_guess):
        """Store initial guess for parameters."""
        sorted_guess = sorted(initial_guess.items(), key=lambda p: p[0])
        self.initial_guess = collections.OrderedDict(sorted_guess)

    def set_evaluation_func(self, func):
        """Evaluation func accepts parameter dict and returns score."""
        self.evaluation_func = func

    def set_checkpoint_dir(self, dirname):
        """Store evolution checkpoint in this directory."""
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        self.checkpoint_dir = dirname
        self.checkpoint_fname = join(self.checkpoint_dir, 'evolution_state.pickle')

    def init_individual(self, individual_type):
        """Initialize specimen."""
        individual = individual_type()
        if self.initial_guess is None:
            # initialize randomly
            for possible_values in self.param_grid.values():
                individual.append(random.randrange(0, len(possible_values)))
        else:
            # use closest parameters to initial guess
            for param, possible_values in self.param_grid.items():
                is_str_param = isinstance(possible_values[0], str)
                guess = self.initial_guess[param]
                best_idx = 0

                if not is_str_param:
                    best_dist = abs(possible_values[0] - guess)
                for idx, value in enumerate(possible_values):
                    if is_str_param:
                        if value == guess:
                            best_idx = idx
                            break
                    else:
                        dist = abs(value - guess)
                        if dist < best_dist:
                            best_dist = dist
                            best_idx = idx

                logger.info('Initial value %r for %r', possible_values[best_idx], param)
                individual.append(best_idx)

        return individual

    def mutate_individual(self, individual, indpb):
        """Randomly choose new parameters from the param_grid."""
        for i, possible_values in enumerate(self.param_grid.values()):
            chance = random.random()
            if chance > indpb:
                continue

            chance = random.random()
            if chance > 0.5:
                # random value from a list of possible values
                individual[i] = random.randrange(0, len(possible_values))
            else:
                # choose next value randomly, but somewhat close to the current one
                current_idx = individual[i]
                sigma = len(possible_values) * 0.1
                new_idx = int(np.random.normal(loc=current_idx, scale=sigma))
                new_idx = max(new_idx, 0)
                new_idx = min(new_idx, len(possible_values) - 1)
                individual[i] = new_idx
        return individual,

    def crossover(self, ind1, ind2, indpb):
        """Mate two individuals."""
        for i in range(len(self.param_grid)):
            chance = random.random()
            if chance > indpb:
                continue

            if ind1[i] <= ind2[i]:
                ind1[i] = random.randint(ind1[i], ind2[i])
                ind2[i] = random.randint(ind1[i], ind2[i])
            else:
                ind1[i] = random.randint(ind2[i], ind1[i])
                ind2[i] = random.randint(ind2[i], ind1[i])

        return ind1, ind2

    def individual_to_params(self, individual):
        """Individual is just a set of indices. This function turns indivuduals into param dict."""
        p = {}
        idx = 0
        for param, possible_values in self.param_grid.items():
            p[param] = possible_values[individual[idx]]
            idx += 1
        return p

    def evaluate(self, individual):
        """Evaluate individual, or return from cache if possible."""
        t_individual = tuple(individual)
        verbose = False
        if t_individual in self.cache:
            self.cache_hits += 1
            if verbose:
                logger.info(
                    'Individual %r obtained from cache',
                    self.individual_to_params(individual),
                )
        else:
            self.cache_misses += 1
            if verbose:
                logger.info('CALCULATE %r', self.individual_to_params(individual))
            p = self.individual_to_params(individual)
            val = self.evaluation_func(p)
            self.cache[t_individual] = (val, )
        return self.cache[t_individual]

    def log_halloffame(self):
        """Print current halloffame to log."""
        for idx, high_achiever in enumerate(self.hof):
            score = high_achiever.fitness.values[0]
            params = self.individual_to_params(high_achiever)
            logger.info('Top individual #%d', idx)
            logger.info('fitness: %r, params %r', score, params)

    def save_checkpoint(self):
        """Self state of evolutionary process if checkpoint dir specified."""
        try:
            backup_checkpoint_fname = self.checkpoint_fname + '.backup'
            tmp_checkpoint_fname = self.checkpoint_fname + '.tmp'

            data = {
                'generation': self.generation,
                'population': self.population,
                'cache': self.cache,
                'hof': self.hof,
                'logbook': self.logbook,
            }
            with open(tmp_checkpoint_fname, 'wb') as checkpoint:
                pickle.dump(data, checkpoint)
            if os.path.isfile(self.checkpoint_fname):
                shutil.copyfile(self.checkpoint_fname, backup_checkpoint_fname)
            shutil.copyfile(tmp_checkpoint_fname, self.checkpoint_fname)
        except Exception as exc:
            logger.error('Checkpoint saving failed %r', exc)


    def try_initialize_from_checkpoint(self):
        """Load evolutionary state from file."""
        try:
            if not os.path.isfile(self.checkpoint_fname):
                logger.info('Could not find checkpoint file %s!', self.checkpoint_fname)
                return False

            logger.info('Restoring...')
            data = None
            with open(self.checkpoint_fname, 'rb') as checkpoint:
                data = pickle.load(checkpoint)

            self.generation = data['generation']
            self.population = data['population']
            self.cache = data['cache']
            self.hof = data['hof']
            self.logbook = data['logbook']
            self.initialized = True
        except Exception as exc:
            logger.info('Could not restore %r', exc)
            return False
        else:
            logger.info('Restored successfully!')
            return True

    def get_cache_size(self, _): return len(self.cache)
    def get_cache_hits(self, _): return self.cache_hits
    def get_cache_misses(self, _): return self.cache_misses

    def optimize(
            self, mutate_p=0.3, mate_p=0.3, population_size=12
    ):
        """Run evolutionary algorithm, return best set of params found."""
        toolbox = base.Toolbox()
        toolbox.register('individual', self.init_individual, creator.Individual)
        toolbox.register('population', tools.initRepeat, list, toolbox.individual)

        toolbox.register('mate', self.crossover, indpb=0.3)
        toolbox.register('mutate', self.mutate_individual, indpb=0.3)
        toolbox.register('select', tools.selTournament, tournsize=3)
        toolbox.register('evaluate', self.evaluate)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register('avg', np.nanmean)
        stats.register('min', np.nanmin)
        stats.register('max', np.nanmax)
        stats.register('std', np.nanstd)
        stats.register('cache_size', self.get_cache_size)
        stats.register('cache_hits', self.get_cache_hits)
        stats.register('cache_misses', self.get_cache_misses)

        if not self.initialized:
            logger.info('Initializing from scratch!')
            self.population = toolbox.population(n=population_size)
            self.hof = tools.HallOfFame(3)
            self.logbook = tools.Logbook()
            self.logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
            self.initialized = True

        hist = tools.History()
        toolbox.decorate('mate', hist.decorator)
        toolbox.decorate('mutate', hist.decorator)
        hist.update(self.population)

        self.save_checkpoint()

        # evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in self.population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        self.hof.update(self.population)

        record = stats.compile(self.population) if stats else {}
        self.logbook.record(gen=self.generation, nevals=len(invalid_ind), **record)
        logger.info(self.logbook.stream)

        logger.info('Begin the generational process!')
        self.save_checkpoint()
        while True:
            self.generation += 1
            # select the next generation individuals
            offspring = toolbox.select(self.population, len(self.population))

            # vary the pool of individuals
            offspring = algorithms.varAnd(offspring, toolbox, mate_p, mutate_p)

            # evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # update the hall of fame with the generated individuals
            self.hof.update(offspring)
            self.log_halloffame()

            # Replace the current population by the offspring
            self.population[:] = offspring

            # Append the current generation statistics to the logbook
            record = stats.compile(self.population) if stats else {}
            self.logbook.record(gen=self.generation, nevals=len(invalid_ind), **record)
            logger.info(self.logbook.stream)
            self.cache_hits = self.cache_misses = 0

            self.save_checkpoint()

        return self.individual_to_params(self.hof[0])


def demo_func(par):
    """Test function to optimize."""
    x = par['x']
    y = par['y']
    z = par['z']
    p = par['p']
    s = par['str']
    funcs = {
        'sin': math.sin,
        'cos': math.cos,
    }
    return (x + (-y) * z) / ((funcs[s](p) ** 2) + 1)

def test_hyperopt():
    """Test optimizer on a very simple example."""
    param_grid = {
        'x': np.linspace(-10, 3, num=100),
        'y': np.logspace(-3, 3, num=100),
        'z': [0, -14, 42],
        'p': np.linspace(1, 5, num=100000),
        'str': ('sin', 'cos'),
    }
    initial_guess = {
        'x': 0,
        'y': 100,
        'z': 1,
        'p': 2,
        'str': 'cos',
    }

    hyperopt = EvolutionaryHyperopt()
    hyperopt.set_param_grid(param_grid)
    hyperopt.set_initial_guess(initial_guess)
    hyperopt.set_evaluation_func(demo_func)
    hyperopt.set_checkpoint_dir('.hyperopt.test')
    hyperopt.try_initialize_from_checkpoint()

    try:
        best = hyperopt.optimize()
        logger.info(best)
    except KeyboardInterrupt:
        logger.info('Terminated!')
        hyperopt.log_halloffame()


def main():
    """Script entry point."""
    init_logger(os.path.basename(__file__))
    np.set_printoptions(precision=3)

    test_hyperopt()

    return 0


if __name__ == '__main__':
    sys.exit(main())
