#! /usr/bin/env python
# -*- coding: utf-8 -*-

import math
import json
import copy
import time
import random
import shapely
import collections
import numpy as np
from tqdm import tqdm
from pprint import pprint
import multiprocessing as mp
from os.path import isfile, join



from get_inputs import get_inputs, \
    create_dir, \
    get_params, \
    get_args, \
    get_schattr

from utils import initialize, \
    gen_solutions, \
    get_partition, \
    get_alg_params, \
    make_move, \
    get_neighbors

from functions import update_property, \
    set_params, \
    obj_func, \
    find_change, \
    parameters


def stop_shc(it, stagnate):
    """Checks if the exit condition is satisfied or not"""
    termination = None
    condition = True

    min_stag = 3
    global max_iter

    if stagnate > min_stag:
        termination = 'Stagnation'
        condition = False

    elif it >= max_iter:
        condition = False
        termination = 'MaxIter reached'

    return condition, termination


def stoc_hill_climb(run, *argv):
    (seed, initial, initialization, args, sch, sy, district) = argv

    '''A plan/ configuraion of N areas grouped into M regions, M < N'''
    regions, regids, best_fval = initial['Regions'], initial['RegionIds'], initial['FuncVal']

    args = list(args)
    args.append(regions)
    args.append(regids)
    args = tuple(args)

    print('Run : {} Init Funcval: {:.4f}'.format(run, best_fval))

    swaps, stagnate, iteration = 0, 0, 0
    condition = True
    random.seed(seed)
    start = time.time()

    while condition:
        ''' Step 2. Make a list of M regions '''
        region_list = [x for x in regions.keys()]
        random.shuffle(region_list)

        while len(region_list) > 0:
            ''' Step 3. Select and remove any region K at random from this list '''
            recipient = region_list[random.randrange(len(region_list))]
            region_list.remove(recipient)  # remove it so that new regions can be picked

            improve = True
            while improve:

                improve = False
                ''' Step 4. Identify a set of zones bordering on areas of region K that could be moved
                into region K without destroying the internal contiguity of the donor region(s). '''
                areas = get_neighbors(recipient, args)

                ''' Step 5. Randomly select zones from this list until either there is a local improvement
                in the current value of the objective function or a move that is equivalently as good as 
                the current best. '''
                while not improve and len(areas) > 0:
                    # Search the local solutions of 'recipient' region
                    area = areas[random.randrange(len(areas))]
                    donor = regids[area]  # Define the source cluster
                    areas.remove(area)  # remove it

                    # Compute the objfunc before and after switch
                    change, possible = find_change([donor, recipient], area, args)
                    """Then make the move, update the list of candidate zones, and return to Step 4"""
                    if possible and change <= 0:

                        moved = make_move([donor, recipient], area, args)
                        if moved:
                            swaps += 1

                    '''Else repeat step 5 until the list is exhausted.'''

            '''Step 6. When the list for region K is exhausted return to step 3, select another region,
            and repeat steps 4-6 .'''

        # Updates
        iteration += 1
        fval = obj_func(regions.keys(), regions)

        if fval >= best_fval:
            stagnate += 1
        else:
            best_fval = fval
            stagnate = 0
        '''
        if iteration == 1 or iteration % 10 == 0:
            print('Run {:<4} Iter {:<4} Objfunc: {:.4f} Stagnation : {:<4} Swaps: {:<4}'.format(run,
                                                                                                iteration,
                                                                                                best_fval,
                                                                                                stagnate,
                                                                                                swaps)
                  )
        '''
        condition, termination = stop_shc(iteration, stagnate)
        if not condition:
            print('Terminating run {}: {}'.format(run, termination))

    telapsed = (time.time() - start) / 60.0  # measures in minutes
    final = get_partition(regions, regids)

    prop, info = get_alg_params(run,
                                args,
                                iteration,
                                telapsed,
                                termination,
                                seed,
                                initialization,
                                sch,
                                district,
                                initial,
                                final)
    t = {'properties': prop, 'info': info}
    return run, t


def stop_SA(it, stagnate):
    """Checks if the exit condition is satisfied or not"""
    termination = None
    condition = True

    min_stag = 3
    repetition = 5
    global partitions, max_iter

    if stagnate > min_stag:
        termination = 'Stagnation'
        condition = False

    elif it >= max_iter:
        condition = False
        termination = 'MaxIter reached'

    elif partitions.count(partitions[-1]) > repetition:
        condition = False
        termination = 'Repetition'

    return condition, termination


def sim_anneal(run, *argv):
    (seed, initial, initialization, args, sch, sy, district) = argv

    ''' A plan/configuraion of N areas grouped into M regions, M < N'''
    regions, regids, best_fval = initial['Regions'], initial['RegionIds'], initial['FuncVal']

    args = list(args)
    args.append(regions)
    args.append(regids)
    args = tuple(args)

    print('Run : {} Init Funcval: {:.4f}'.format(run, best_fval))
    final = copy.deepcopy(initial)

    swaps, stagnate, iteration = 0, 0, 0
    condition = True
    random.seed(seed)

    global partitions
    partitions = list()
    a, T, T0, minQ = get_SAparams()  # SA parameters

    start = time.time()
    while condition:
        Q = 0

        region_list = [x for x in regions.keys()]
        random.shuffle(region_list)

        while len(region_list) > 0:

            recipient = region_list[random.randrange(len(region_list))]
            region_list.remove(recipient)  # remove it so that new regions can be picked

            improve = True
            while improve:
                areas = get_neighbors(recipient, args)
                improve = False

                while not improve and len(areas) > 0:
                    # Randomly select zones until there is a local improvement
                    area = areas[random.randrange(len(areas))]
                    donor = regids[area]  # Define the source cluster

                    # Compute the objfunc before and after switch
                    change, possible = find_change([donor, recipient], area, args)

                    if possible and change is not None:
                        if change <= 0:
                            moved = make_move([donor, recipient], area, args)
                            if moved:
                                swaps += 1

                        # Else make the Simulated Annealing move with Boltzmann's probability
                        elif Q < minQ and T > T0 and random.random() < np.exp(-change / T):
                            if make_move([donor, recipient], area, args):
                                Q += 1
                                # break

                    areas.remove(area)  # remove it

        # Updates
        T = update(a,
                   T,
                   iteration,
                   1)  # fast annealing
        # T = update(a, T0, iteration, 2)  # slow annealing

        iteration += 1
        fval = obj_func(regions.keys(), regions)
        partitions.append(tuple(regids.values()))

        if fval >= best_fval:
            stagnate +=1
        else:
            final = get_partition(regions,
                                  regids)
            best_fval = fval
            stagnate = 0
        '''
        if iteration == 1 or iteration % 10 == 0:
            print('Run {:<4} Iter {:<4} Objfunc: {:.5f} Stagnation : {:<4}'.format(run,
                                                                                   iteration,
                                                                                   best_fval,
                                                                                   stagnate)
                  )
        '''
        condition, termination = stop_SA(iteration, stagnate)
        if not condition:
            print('Terminating run {}: {}'.format(run, termination))

    telapsed = (time.time() - start) / 60.0  # measures in minutes

    prop, info = get_alg_params(run,
                                args,
                                iteration,
                                telapsed,
                                termination,
                                seed,
                                initialization,
                                sch,
                                district,
                                initial,
                                final)
    t = {'properties': prop, 'info': info}
    return run, t


def get_SAparams():
    """Define SA parameters"""
    T = 1
    Q = 10

    a = 0.85
    T0 = 0

    return a, T, T0, Q


def update(at, T, k, option):
    """ Modifies the temperature of annealing process """
    if option == 1:  # fast annealing
        T = at * T  # T(k) = a.T(k-1)

    elif option == 2:  # gradual annealing
        T = T / (1 + math.log(k))  # T(k) = T(0)/ln(k)

    return T


""" Code for Tabu Search"""

Move = collections.namedtuple("Move", "area donor recipient ")


def stop_tabu(iterations, stagnate):
    """Checks if the exit condition is satisfied or not"""
    termination = None
    condition = True

    min_stag = 10
    repetition = 5
    global partitions, max_iter

    if stagnate > min_stag:
        termination = 'Stagnation'
        condition = False

    elif iterations >= max_iter:
        condition = False
        termination = 'MaxIter reached'

    elif partitions.count(partitions[-1]) > repetition:
        condition = False
        termination = 'Repetition'

    return condition, termination


def init_list():
    R = 80  # length of tabu list
    move_list = collections.deque([], R)
    return move_list


def update_tabulist(moved):
    """Moving polygon between regions and update tabu_list"""

    global tabu_list, Move

    # Update the tabu_list with the reverse move
    reverse_move = Move(moved.area, moved.recipient, moved.donor)    # ("Move", "area donor recipient ")
    tabu_list.append(reverse_move)

    # forward_move = Move(moved.area, moved.donor, moved.recipient)
    # tabu_list.append(forward_move)

from concurrent.futures import ThreadPoolExecutor, as_completed


def check(area, donor, recipient, args):
    possible_move = Move(area, donor, recipient)
    status = None    # 1- not forbidden, 0 - forbidden/tabu
    change = None
    global tabu_list

    # Save the global best if not prohibited by Tabu
    if possible_move not in tabu_list:
        change, possible = find_change([donor, recipient], area, args)
        if possible:
            status = 1
    else:
        change, possible = find_change([donor, recipient], area, args)
        if possible:
            status = 0

    return possible_move, status, change


def find_best_move(region_list, args):

    regids = args[8]
    best_move = None
    best_diff = float("inf")
    improving_tabus = list()

    while region_list:
        # Pick a region and search for neighboring areas that can be swapped w/o breaking contiguity
        recipient = region_list[random.randrange(len(region_list))]
        region_list.remove(recipient)  # remove it so that new regions can be picked
        areas = get_neighbors(recipient, args)

        # Trying a parallel version of the search for best solution
        with ThreadPoolExecutor() as executor:

            results = {executor.submit(check,
                                       area,
                                       regids[area],
                                       recipient,
                                       args): area for area in areas}

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


def tabu_search(run, *argv):

    (seed, initial, initialization, args, sch, sy, district) = argv

    # A plan/configuration of N areas grouped into M regions, M < N
    regions, regids, best_fval = initial['Regions'], initial['RegionIds'], initial['FuncVal']

    args = list(args)
    args.append(regions)
    args.append(regids)
    args = tuple(args)

    print('Run : {} Init Funcval: {:.4f}'.format(run, best_fval))
    final = copy.deepcopy(initial)

    stagnate, iteration = 0, 0
    condition = True
    random.seed(seed)

    global tabu_list, partitions
    partitions, tabu_list = list(), init_list()

    start = time.time()
    while condition:
        # Make a list of M regions
        region_list = [x for x in regions.keys()]
        random.shuffle(region_list)

        # Parallel search for best move
        best_move, best_diff, improving_tabus = find_best_move(region_list, args)
        moved = False

        # IMPROVING MOVE
        if best_move is not None and best_diff <= 0:
            moved = make_move(
                [
                    best_move.donor,
                    best_move.recipient
                ],
                best_move.area,
                args
            )
            if moved:
                update_tabulist(best_move)

        # ASPIRATION MOVE
        if not moved:
            if improving_tabus:

                aspiration_move = improving_tabus[random.randrange(len(improving_tabus))]
                moved = make_move(
                    [
                        aspiration_move.donor,
                        aspiration_move.recipient
                    ],
                    aspiration_move.area,
                    args)

                if moved:
                    update_tabulist(aspiration_move)

        # NEITHER IMPROVEMENT NOR ASPIRATION
        if not moved:

            if best_move is not None:
                moved = make_move(
                    [
                        best_move.donor,
                        best_move.recipient
                    ],
                    best_move.area,
                    args
                )

                if moved:
                    update_tabulist(best_move)

        # Updates
        iteration += 1
        fval = obj_func(regions.keys(), regions)
        partitions.append(tuple(regids.values()))

        if fval >= best_fval:
            stagnate += 1
        else:
            final = get_partition(regions, regids)
            best_fval = fval
            stagnate = 0
        '''
        if iteration == 1 or iteration % 10 == 0:
            print(' Run {:<4} Iter {:<4} Objfunc: {:.4f}  Stagnation : {:<4}'.
                  format(run,
                         iteration,
                         best_fval,
                         stagnate)
                  )
        '''
        # Termination
        condition, termination = stop_tabu(iteration, stagnate)
        if not condition:
            print('Terminating run {}: {}'.format(run, termination))

    telapsed = (time.time() - start) / 60.0  # measures in minutes

    prop, info = get_alg_params(run,
                                args,
                                iteration,
                                telapsed,
                                termination,
                                seed,
                                initialization,
                                sch,
                                district,
                                initial,
                                final)
    t = {'properties': prop, 'info': info}

    return run, t


def make_runs(options):
    """ Simulate runs for the local search methods """
    year = options.year
    seed = options.seed
    sch = options.school
    num_runs = options.runs
    num_process = min(options.processes, num_runs)
    district = options.district
    init = options.initialization  # 1: 'seeded', 2: 'regional', 3: 'existing', 4: 'random'

    read_path = "../{}/data/".format(district)
    sy = "{}_{}".format(year, (year + 1) % 100)

    functions = {'SHC': stoc_hill_climb,
                 'SA': sim_anneal,
                 'TS': tabu_search
                 }
    spas, \
    spas_nbr, \
    sch_spas_nbr, \
    adjacency, \
    popltn, \
    capacity = get_inputs(sch, sy, read_path, district)
    schlattr = get_schattr(sch, district)
    arg = get_args(popltn,
                   capacity,
                   adjacency,
                   spas,
                   spas_nbr,
                   sch_spas_nbr,
                   schlattr)

    """Seeding ensures starting configurations are consistent"""
    random.seed(seed)
    seeds = [x + random.randrange(100000) for x in range(num_runs)]
    starting = {1: 'seeded', 2: 'regional', 3: 'existing', 4: 'infeasible'}
    solutions = gen_solutions()  # holds starting solutions

    global max_iter
    # Set weight
    w = (8, 7)[district=='B']
    set_params(w)
    w1, w2, epsilon, max_iter = parameters()

    print('Generating starting solutions!!\n\n')
    solutions = gen_solutions(arg, i=init, runs=num_runs, seeds=seeds)

    # methods = [options.algo]  # To run single algorithm, default being SHC. See main()
    methods = [h for h in functions.keys()]  # To run all three methods

    for algo in methods:
        if solutions:
            print('\n\n Running local search {}'.format(algo))
            try:
                # Parallel (asynchronous) version
                pool = mp.Pool(processes=num_process)
                output = [
                    pool.apply_async(functions[algo],
                                     args=(r + 1,
                                           seeds[r],
                                           solutions[r],
                                           starting[init],
                                           arg,
                                           sch,
                                           sy,
                                           district
                                           )
                                     )
                    for r in range(num_runs)
                ]
                pool.close()
            except Exception as e:
                print(e)
                continue
        else:
            print('Couldn\'t run local search. Skipping!!')
            continue

        run_results = dict()
        results = [p.get() for p in output]

        for r, result in results:
            run_results[r] = result

        write_path = create_dir(district,
                                'results',
                                algo,
                                "SY{}".format(sy),
                                starting[init],
                                sch)

        with open(join(write_path,
                       "run{}_{}_{}.json".format(int(w),
                                                 sch,
                                                 algo)
                       ), 'w') as outfile:
            json.dump(run_results, outfile)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-q", "--quiet", action="store_false", dest="verbose",
                        help="don't print status messages to stdout")
    parser.add_argument("-s", "--school", type=str, default="ES")   # school: Elementary-ES Middle-MS High-HS
    parser.add_argument("-a", "--algo", type=str, default="SHC")   # algorithm
    parser.add_argument("-i", "--initialization", default=1, type=int)
    parser.add_argument("-d", "--district", type=str, default="B")  # district: A or B

    parser.add_argument("-p", "--processes", type=int, default=mp.cpu_count() - 1)  # number of parallel processes
    parser.add_argument("-r", "--runs", type=int, default=51)  # number of runs
    parser.add_argument("-y", "--year", type=int, default=2017)  # school year
    parser.add_argument("-e", "--seed", type=int, default=17)    # integer seed for random number generator

    options = parser.parse_args()
    make_runs(options)

    return 0


if __name__ == "__main__":
    import sys

    print(sys.platform)
    sys.exit(main())
