#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import math
import json
import copy
import time
import random
import argparse
import collections
import numpy as np
import multiprocessing as mp
from os.path import isfile, join
from concurrent.futures import ThreadPoolExecutor, as_completed

from get_inputs import GetInputs

from utils import Utils

from functions import update_property, \
    set_params, \
    obj_func, \
    find_change, \
    parameters


class REGAL:
    """
    The base class for the REGAL (Regionalization through locAL search) algorithm. For more details refer to the paper:
    Biswas S, Chen F, Chen Z, Sistrunk A, Self N, Lu CT, Ramakrishnan N. REGAL: A zonealization framework for school
    boundaries. InProceedings of the 27th ACM SIGSPATIAL International Conference on Advances in Geographic Information
    Systems 2019 Nov 5 (pp. 544-547).
    """

    def __init__(self, args=None, solution=None):
        self.args = args
        self.solution = solution
        self.max_iter = parameters()[3]


class SHC(REGAL):
    """
    REGAL framework with Stochastic Hill Climbing as the local search algorithm.
    """

    def __init__(self, args=None, solution=None, seed=None):
        """
        Constructor to initialize default values.
        """
        super().__init__(args, solution)
        self.seed = seed

        # Hyperparameters
        self.min_stag = 3

    def stop_algo(self):
        """
        Checks if the exit condition is satisfied or not.
        """
        termination = None
        condition = True

        if self.stagnate_count > self.min_stag:
            termination = 'Solution stagnation'
            condition = False

        elif self.iter >= self.max_iter:
            condition = False
            termination = 'Maximum iterations reached'

        return condition, termination

    def search(self, run, argv):
        """
        Local search: stochastic hill climbing
        """
        (initialization, sch, sy) = argv

        ''' 1. A plan/ zoning of N areas grouped into K zones, K < N'''
        initial = copy.deepcopy(self.solution)
        zones, zoneids, best_fval = self.solution['zones'], self.solution['zoneIds'], self.solution['FuncVal']
        print('Run : {} Init Funcval: {:.4f}'.format(run, best_fval))

        args = list(self.args)
        args.append(zones)
        args.append(zoneids)
        self.args = tuple(args)

        switches, self.stagnate_count, self.iter = 0, 0, 0
        condition, termination = True, 'Error!!'
        random.seed(self.seed)

        # Start the search process
        start = time.time()
        util = Utils(args)

        while condition:
            ''' 2. Make a list of M zones '''
            zone_list = [x for x in zones.keys()]
            random.shuffle(zone_list)

            while len(zone_list) > 0:
                ''' 3. Select and remove any zone k at random from this list '''
                recipient = zone_list[random.randrange(len(zone_list))]
                zone_list.remove(recipient)  # remove it so that new zones can be picked

                improve = True
                while improve:

                    improve = False
                    ''' 4. Identify a set of zones bordering on areas of zone K that could be moved into zone K without
                     destroying the internal contiguity of the donor zone(s). '''
                    areas = util.get_neighbors(recipient, self.args)

                    '''5. Randomly select zones from this list until either there is a local improvement in the current
                     value of the objective function or a move that is equivalently as good as the current best. '''
                    while not improve and len(areas) > 0:
                        # Search the local solutions of 'recipient' zone
                        area = areas[random.randrange(len(areas))]
                        donor = zoneids[area]  # Define the source cluster
                        areas.remove(area)  # remove it

                        # Compute the objfunc before and after switch
                        change, possible = find_change([donor, recipient], area, self.args)
                        # Then make the move, update the list of candidate zones, and return to 4
                        if possible and change <= 0:
                            moved = util.make_move([donor, recipient], area, self.args)
                            if moved:
                                switches += 1
                            if change < 0:
                                improve = True

                        # Else repeat step 5 until the list is exhausted.

                # 6. When the list for zone K is exhausted return to step 3, select another zone and repeat steps 4-6.

            # Updates
            self.iter += 1
            fval = obj_func(zones.keys(), zones)

            if fval >= best_fval:
                self.stagnate_count += 1
            else:
                best_fval = fval
                self.stagnate_count = 0

            if self.iter == 1 or self.iter % 10 == 0:
                print('Run {:<4} Iter {:<4} Objfunc: {:.2f} Stagnation: {:<4} Switches: {:<4}'
                      ' Time: {:<4} min'.format(run, self.iter, best_fval, self.stagnate_count, switches,
                                                int(1 + (time.time() - start) / 60)
                                                )
                      )
            condition, termination = self.stop_algo()
            if not condition:
                print('Terminating run {}: {}'.format(run, termination))

        telapsed = (time.time() - start) / 60.0  # measures in minutes

        final = util.get_partition(zones, zoneids)

        prop, info = util.get_alg_params(run, args, self.iter, telapsed, termination, self.seed,
                                         initialization, sch,
                                         initial, final)
        t = {'properties': prop, 'info': info}
        return run, t


class SA(REGAL):
    """
    REGAL framework with Stochastic Hill Climbing as the local search algorithm.
    """

    def __init__(self, args=None, solution=None, seed=None):
        """
        Constructor to initialize default values.
        """
        super().__init__(args, solution)
        self.seed = seed
        self.partitions = list()

        # Hyper-parameters of Simulated Annealing
        self.T = 1
        self.T0 = 0
        self.a = 0.85
        self.minQ = 10

        # Parameters for checking termination condition
        self.min_stag = 3
        self.repetition = 5

    def update(self, option):
        """ Modifies the temperature of annealing process """
        if option == 1:  # fast annealing
            self.T *= self.a  # T(k) = a.T(k-1)

        elif option == 2:  # gradual annealing
            self.T /= (1 + math.log(self.iter))  # T(k) = T(0)/ln(k)

    def stop_algo(self):
        """
        Checks if the exit condition is satisfied or not.
        """
        termination, condition = None, True

        if self.stagnate_count > self.min_stag:
            termination = 'Solution stagnation'
            condition = False

        elif self.iter >= self.max_iter:
            condition = False
            termination = 'Maximum iterations reached'

        elif self.partitions.count(self.partitions[-1]) > self.repetition:
            condition = False
            termination = 'Repetition of solutions'

        return condition, termination

    def search(self, run, argv):
        """
        Local search: Simulated Annealing
        """
        (initialization, sch, sy) = argv

        ''' 1. A plan/ zoning of N areas grouped into K zones, K < N'''
        initial, final = copy.deepcopy(self.solution), copy.deepcopy(self.solution)
        zones, zoneids, best_fval = self.solution['zones'], self.solution['zoneIds'], self.solution['FuncVal']
        print('Run : {} Init Funcval: {:.4f}'.format(run, best_fval))

        args = list(self.args)
        args.append(zones)
        args.append(zoneids)
        self.args = tuple(args)

        switches, self.stagnate_count, self.iter = 0, 0, 0
        condition, termination = True, 'Error!!'
        random.seed(self.seed)

        # Start the search process
        start = time.time()
        util = Utils(args)

        while condition:
            Q = 0

            zone_list = [x for x in zones.keys()]
            random.shuffle(zone_list)

            while len(zone_list) > 0:

                recipient = zone_list[random.randrange(len(zone_list))]
                zone_list.remove(recipient)  # remove it so that new zones can be picked

                improve = True
                while improve:
                    areas = util.get_neighbors(recipient, self.args)
                    improve = False

                    while not improve and len(areas) > 0:
                        # Randomly select zones until there is a local improvement
                        area = areas[random.randrange(len(areas))]
                        donor = zoneids[area]  # Define the source cluster

                        # Compute the objfunc before and after switch
                        change, possible = find_change([donor, recipient], area, self.args)

                        if possible and change is not None:
                            if change <= 0:
                                moved = util.make_move([donor, recipient], area, self.args)
                                if moved:
                                    switches += 1
                                if change < 0:
                                    improve = True

                            # Else make the Simulated Annealing move with Boltzmann's probability
                            elif Q < self.minQ and self.T > self.T0 and random.random() < np.exp(-change / self.T):
                                if util.make_move([donor, recipient], area, self.args):
                                    Q += 1

                        areas.remove(area)  # remove it

            # Updates
            self.update(1)  # fast annealing
            # self.update(2)  # slow annealing
            self.iter += 1
            self.partitions.append(tuple(zoneids.values()))

            fval = obj_func(zones.keys(), zones)

            if fval >= best_fval:
                self.stagnate_count += 1
            else:
                final = util.get_partition(zones, zoneids)
                best_fval = fval
                self.stagnate_count = 0

            if self.iter == 1 or self.iter % 10 == 0:
                print('Run {:<4} Iter {:<4} Objfunc: {:.2f} Stagnation: {:<4} Switches: {:<4} Q: {}'
                      ' Time: {:<4} min'.format(run, self.iter, best_fval, self.stagnate_count, switches, Q,
                                                int(1 + (time.time() - start) / 60)
                                                )
                      )
            condition, termination = self.stop_algo()
            if not condition:
                print('Terminating run {}: {}'.format(run, termination))

        telapsed = (time.time() - start) / 60.0  # measures in minutes

        prop, info = util.get_alg_params(run, args, self.iter, telapsed, termination, self.seed,
                                         initialization, sch,
                                         initial, final)
        t = {'properties': prop, 'info': info}
        return run, t


class TS(REGAL):
    """
    REGAL framework with Tabu Search as the local search algorithm.
    """

    Move = collections.namedtuple("Move", "area donor recipient ")

    def __init__(self, args=None, solution=None, seed=None):
        """
        Constructor to initialize default values.
        """
        super().__init__(args, solution)
        self.seed = seed
        self.partitions = list()

        # Hyper-parameters of Tabu search
        R = 80  # length of tabu list
        self.tabu_list = collections.deque([], R)

        # Parameters for checking termination condition
        self.min_stag = 10
        self.repetition = 5

    def stop_algo(self):
        """
        Checks if the exit condition is satisfied or not.
        """
        termination, condition = None, True

        if self.stagnate_count > self.min_stag:
            termination = 'Solution stagnation'
            condition = False

        elif self.iter >= self.max_iter:
            condition = False
            termination = 'Maximum iterations reached'

        elif self.partitions.count(self.partitions[-1]) > self.repetition:
            condition = False
            termination = 'Repetition of solutions'

        return condition, termination

    def update_tabulist(self, moved):
        """Moving areas between zones and update tabu_list"""

        # Update the tabu_list with the reverse move
        reverse_move = self.Move(moved.area, moved.recipient, moved.donor)  # ("Move", "area donor recipient ")
        self.tabu_list.append(reverse_move)

        # forward_move = Move(moved.area, moved.donor, moved.recipient)
        # tabu_list.append(forward_move)

    def check(self, area, donor, recipient, args):
        possible_move = self.Move(area, donor, recipient)
        status = None  # 1- not forbidden, 0 - forbidden/tabu
        change = None

        # Save the global best if not prohibited by Tabu
        if possible_move not in self.tabu_list:
            change, possible = find_change([donor, recipient], area, args)
            if possible:
                status = 1
        else:
            change, possible = find_change([donor, recipient], area, args)
            if possible:
                status = 0

        return possible_move, status, change

    def find_best_move(self, zone_list, util):

        zoneids = self.args[8]
        best_move = None
        best_diff = float("inf")
        improving_tabus = list()

        while zone_list:
            # Pick a zone and search for neighboring areas that can be swapped w/o breaking contiguity
            recipient = zone_list[random.randrange(len(zone_list))]
            zone_list.remove(recipient)  # remove it so that new zones can be picked
            areas = util.get_neighbors(recipient, self.args)

            # Trying a parallel version of the search for best solution
            with ThreadPoolExecutor() as executor:

                results = {executor.submit(self.check,
                                           area,
                                           zoneids[area],
                                           recipient,
                                           self.args): area for area in areas}

                for f in as_completed(results):
                    possible_move, status, change = f.result()
                    if status is not None:
                        try:
                            if status == 1 and change < best_diff:
                                best_move = possible_move
                                best_diff = change

                            elif status == 0 and change < 0:
                                improving_tabus.append(possible_move)

                            else:
                                # print('Not a legit move')
                                pass

                        except Exception as e:
                            print(e)

        return best_move, best_diff, improving_tabus

    def search(self, run, argv):
        """
        Local search: Tabu Search
        """
        (initialization, sch, sy) = argv

        ''' 1. A plan/ zoning of N areas grouped into K zones, K < N'''
        initial, final = copy.deepcopy(self.solution), copy.deepcopy(self.solution)
        zones, zoneids, best_fval = self.solution['zones'], self.solution['zoneIds'], self.solution['FuncVal']
        print('Run : {} Init Funcval: {:.4f}'.format(run, best_fval))

        args = list(self.args)
        args.append(zones)
        args.append(zoneids)
        self.args = tuple(args)

        self.stagnate_count, self.iter = 0, 0
        condition, termination = True, 'Error!!'
        random.seed(self.seed)

        # Start the search process
        start = time.time()
        util = Utils(args)

        while condition:
            # Make a list of M zones
            zone_list = [x for x in zones.keys()]
            random.shuffle(zone_list)

            # Parallel search for best move
            best_move, best_diff, improving_tabus = self.find_best_move(zone_list, util)
            moved = False

            # IMPROVING MOVE
            if best_move is not None and best_diff <= 0:
                moved = util.make_move(
                    [
                        best_move.donor,
                        best_move.recipient
                    ],
                    best_move.area,
                    self.args
                )

                if moved:
                    self.update_tabulist(best_move)

            # ASPIRATION MOVE
            if not moved:
                if improving_tabus:

                    aspiration_move = improving_tabus[random.randrange(len(improving_tabus))]
                    moved = util.make_move(
                        [
                            aspiration_move.donor,
                            aspiration_move.recipient
                        ],
                        aspiration_move.area,
                        self.args
                    )

                    if moved:
                        self.update_tabulist(aspiration_move)

            # NEITHER IMPROVEMENT NOR ASPIRATION
            if not moved:

                if best_move is not None:
                    moved = util.make_move(
                        [
                            best_move.donor,
                            best_move.recipient
                        ],
                        best_move.area,
                        self.args
                    )

                    if moved:
                        self.update_tabulist(best_move)

            # Updates
            self.iter += 1
            fval = obj_func(zones.keys(), zones)
            self.partitions.append(tuple(zoneids.values()))

            if fval >= best_fval:
                self.stagnate_count += 1
            else:
                final = util.get_partition(zones, zoneids)
                best_fval = fval
                self.stagnate_count = 0

            if self.iter == 1 or self.iter % 10 == 0:
                print('Run {:<4} Iter {:<4} Objfunc: {:.2f} Stagnation: {:<4}'
                      ' Time: {:<4} min'.format(run, self.iter, best_fval, self.stagnate_count,
                                                int(1 + (time.time() - start) / 60)
                                                )
                      )
            # Termination
            condition, termination = self.stop_algo()
            if not condition:
                print('Terminating run {}: {}'.format(run, termination))

        telapsed = (time.time() - start) / 60.0  # measures in minutes
        prop, info = util.get_alg_params(run, args, self.iter, telapsed, termination, self.seed,
                                         initialization, sch,
                                         initial, final)
        t = {'properties': prop, 'info': info}
        return run, t


def local_search(r, algo, args, solution, seed, *argv):
    """
    Performs search by calling required local search procedures
    """
    try:
        # Creat an object method that calls the respective search procedure
        if algo == "SHC":
            method = SHC(args, solution, seed)
        elif algo == "SA":
            method = SA(args, solution, seed)
        elif algo == "TS":
            method = TS(args, solution, seed)
        else:
            pass

        return method.search(r, argv)

    except Exception as e:
        print(e)


def make_runs(options):
    """
    Simulate runs for the local search methods.
    """
    sch = options.school
    sy = 'SY{}_{}'.format(options.year, (options.year + 1) % 100)
    init = options.initialization
    runs = options.runs

    # Read data files
    inputs = GetInputs(sch)
    args = inputs.get_inputs()  # args = (population, capacity, adjacency, spas, spas_nbr, sch_spas_nbr, schlattr)

    # Seeding ensures starting configurations are consistent
    random.seed(options.seed)
    seeds = [s + random.randrange(1000000) for s in range(runs)]

    # Set weight (w) in the range [0, 1] for calculating F = w * F1 + (1 - w) * F2
    weight = 7
    set_params(weight)

    # Generating starting solutions for each run
    init_type = {1: 'seeded',
                 2: 'infeasible',
                 3: 'existing'}
    print('\n Generating starting solutions!! \n')
    util = Utils(args, seeds)
    solutions = util.gen_solutions(init, runs, args, seeds)

    if solutions:
        for algo in ['SHC', 'SA', 'TS']:
            print('\nRunning local search {}\n'.format(algo))
            try:
                # Parallel (asynchronous) version
                num_proc = min(runs, mp.cpu_count() - 1)
                num_processes = (num_proc, 1 + int(num_proc / 5))[algo == 'TS']  # has parallelization inside it
                pool = mp.Pool(processes=num_processes)

                output = [
                    pool.apply_async(local_search,
                                     args=(r + 1,
                                           algo, args, solutions[r], seeds[r],
                                           init_type[init], sch, sy
                                           )
                                     )
                    for r in range(runs)
                ]
                pool.close()

                # Compiling the results of the run
                results = [p.get() for p in output]
                run_results = dict()
                for r, result in results:
                    run_results[r] = result

                # Writing the results
                write_path = util.create_dir('results',
                                             algo
                                             )
                with open(join(write_path,
                               "run{}_{}_{}.json".format(int(weight),
                                                         sch,
                                                         algo)
                               ), 'w') as outfile:
                    json.dump(run_results, outfile)

            except Exception as e:
                print(e)
    else:
        print('Couldn\'t run local search. Exiting the program!!')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-q", "--quiet", action="store_false", dest="verbose",
                        help="don't print status messages to stdout")
    parser.add_argument("-s", "--school", type=str, default="ES")  # schools: ES, MS, HS
    parser.add_argument("-r", "--runs", type=int, default=25)  # number of runs to be simulated
    parser.add_argument("-e", "--seed", type=int, default=17)  # integer seed for random number generator
    parser.add_argument("-i", "--initialization", default=1, type=int)  # 1: seeded, 2: infeasiible 3: existing
    parser.add_argument("-y", "--year", type=int, default=2019)  # school year
    options = parser.parse_args()
    make_runs(options)

    return 0


if __name__ == "__main__":
    print(sys.platform)
    sys.exit(main())
