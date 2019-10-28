import random
import copy
import time
from tqdm import tqdm
from pprint import pprint
import multiprocessing as mp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from functions import obj_func,\
    update_property,\
    find_change,\
    parameters,\
    target_compact, target_balance
from get_inputs import get_params


def make_move(cids, area, args):
    """Moving polygon between clusters"""

    donor = cids[0]
    recip = cids[1]

    # args = (popltn, capacity, adjacency, spas, spas_nbr, sch_spas_nbr, sch, regions, regids)
    regions = args[7]

    donor_region = [m for m in regions[donor]['STATE']]
    recip_region = [m for m in regions[recip]['STATE']]

    moved = False
    try:
        if len(donor_region) > 1:
            # Make the move
            donor_region.remove(area)
            recip_region.append(area)

            # Update the regions
            regions[donor]['STATE'] = donor_region
            regions[recip]['STATE'] = recip_region

            regids = args[8]
            regids[area] = recip

            update_property(cids, args)
            moved = True

    except Exception as e:
        print(e)
        print('Exception in make_move')

    return moved


def get_alg_params(run, args, *argv):
    # argv = (iteration, t_elapsed, termination, seed, initialization, sch, district, inital, final)
    iteration = argv[0]
    t_elapsed = argv[1]
    terminate = argv[2]
    seed = argv[3]
    initialize = argv[4]
    sch = argv[5]
    district = argv[6]

    """Consolidating all the attributes"""
    # Constants defined as global variables
    w1, w2, epsilon, max_iter = parameters()

    alg_params = get_params(w1=w1,
                            w2=w2,
                            epsilon=epsilon,
                            MaxIter=max_iter)

    params = get_params(AlgParams=alg_params,
                        Iteration=iteration,
                        TimeElapsed=t_elapsed,
                        Termination=terminate,
                        Seed=seed,
                        School=sch,
                        District=district,
                        Initialization=initialize)

    initial = argv[7]
    final = argv[8]
    existing = get_existing(args)

    print("Run: {}\t Iterations: {}\t Time: {:.4f} min\t"
          "  Initial CV: {:.4f} Final CV {:.4f} Existing CV: {:.4f}".format(run,
                                                                            iteration,
                                                                            t_elapsed,
                                                                            initial['FuncVal'],
                                                                            final['FuncVal'],
                                                                            existing['FuncVal'])
          )
    run_info = {'Existing': existing,
                'Initial': initial,
                'Final': final}
    return params, run_info


def get_neighbors(regid, args):
    """Get the list of areas adjacent to the base region"""

    # args = (popltn, capacity, adjacency, spas, spas_nbr, sch_spas_nbr, schlattr, regions, regids)
    sch_spas_nbr = args[5]

    sch_spas = [x for x in sch_spas_nbr.keys()]
    spas_nbr = args[4]
    regions = args[7]
    regids = args[8]

    neighbors = []
    for area in regions[regid]['STATE']:
        neighbors = neighbors + [x for x in spas_nbr[str(area)]
                                 if regids[x] != regid and x not in sch_spas]

    neighbors = list(set(neighbors))  # get unique values

    # Check which areas break contiguity on being swapped
    to_remove = []

    for area in neighbors:

        donor_regid = regids[area]
        donor_region = [m for m in regions[donor_regid]['STATE']]

        if area not in donor_region:
            pass
            # print(donor_regid, ':', donor_cluster)
        else:
            donor_region.remove(area)

        if len(donor_region) > 1:  # If the cluster is not a singleton

            adjacency = args[2]
            donor_region_adj = adjacency.loc[donor_region, donor_region].values
            adjacent_mat = csr_matrix(donor_region_adj)
            num_connect_comp = connected_components(adjacent_mat, directed=False, return_labels=False)

            if num_connect_comp != 1:  # Not 1 means disconnected
                to_remove.append(area)

    # Remove those areas that break contiguity
    for area in to_remove:
        neighbors.remove(area)

    return neighbors


def get_adj_areas(areas, spas_nbr, regids):
    """Returns adjacent unassigned area polygons to a cluster"""
    adj_areas = []
    if len(areas) > 0:
        for area in areas:
            adj_areas = adj_areas + [a for a in spas_nbr[area]
                                     if regids[a] == -1]

        adj_areas = list(set(adj_areas))

    return adj_areas


def get_partition(regions, regids):
    r = copy.deepcopy(regions)
    i = copy.deepcopy(regids)
    f = obj_func(r.keys(), r)

    partition = get_params(Regions=r,
                           RegionIds=i,
                           FuncVal=f)
    return partition


def get_existing(args):

    regions, regids, dummy = initialize(3, args)
    update_property(regions.keys(), args, regions)
    existing = get_partition(regions, regids)

    return existing


def seeded_init(args, seed=None):
    """ Seeded initialization starting with school containing polygons """

    random.seed(seed)
    regions = dict()
    regids = dict()

    # args = (popltn, capacity, adjacency, spas, spas_nbr, sch_spas_nbr, schlattr, regions, regids)
    spas, spas_nbr, sch_spas_nbr, schlattr = args[3], args[4], args[5], args[6]

    # Enumerate regions and set status
    spas_list = [s for s in spas_nbr.keys()]
    for area in spas_list:
        regids[area] = -1  # -1 means not assigned to a region

    # Initialize the regions with school-containing polygons
    sch_list = []
    for area in sch_spas_nbr.keys():
        sch_code = spas[schlattr][area]
        regids[area] = sch_code  # Assign the SCH CODE to the area

        spas_list.remove(area)      # Remove areas that have already been assigned to a region
        sch_list.append(sch_code)

        regions[sch_code] = get_params(STATE=[area], SCH=sch_code)

    num_regions = len(sch_list)     # No. of schools

    while len(spas_list) > 0:
        # Pick a random region
        regid = sch_list[random.randrange(num_regions)]
        region = [x for x in regions[regid]['STATE']]
        adj_areas = get_adj_areas(region, spas_nbr, regids)   # Get list of free areas around it

        if len(adj_areas) > 0:
            area = adj_areas[random.randrange(len(adj_areas))]
            regids[area] = regid
            regions[regid]['STATE'].append(area)
            spas_list.remove(area)

    if list(regids.values()).count(-1) > 0:
        # print('There are unassigned polygons present. Error!!')
        pass

    return regions, regids


def par_check(r, a, region, args):
    """Parallel checking local moves"""

    members = [m for m in region] + [a]
    random.shuffle(members)

    possible = True
    final = 1

    # Find the population capacity balance
    f1 = target_balance(members=members,
                        pop=args[0],
                        cap=args[1]
                        )[2]
    # Find compactness score
    f2 = target_compact(members=members,
                        shapes=args[3]
                        )[2]

    if f2 == 1:
        print('Error in moving {} to {}'.format(a, r))
        possible = False
    else:
        w1, w2, epsilon, max_iter = parameters()
        final = w1 * f1 + w2 * f2
        '''final = f2    # compact initialization'''

    return a, final, possible


def find_trials(ids, trials=dict(), inverse=dict(), *argt):

    regions, regids, args = argt
    N = 5   # parameter of the region growth procedure

    for r in ids:

        # For each region, survey the possible nearby moves
        region = [m for m in regions[r]['STATE']]
        neighbors = get_adj_areas(region, args[4], regids)

        change = dict()
        if neighbors:
            initial = regions[r]['F']

            # Using ThreadPoolExecutor from concurrent.futures
            with ThreadPoolExecutor() as executor:
                # Start the load operations and mark each future with its URL
                results = {executor.submit(par_check,
                                           r,
                                           area,
                                           region,
                                           args): area for area in neighbors}

                for f in as_completed(results):

                    area, final, possible = f.result()
                    if possible:
                        try:
                            change[area] = final - initial
                        except Exception as e:
                            print(e)

        # Select the top N plans
        if change:
            try:
                top = [
                    (a, change[a])
                    for a in sorted(change,
                                    key=change.__getitem__
                                    )[:min(N, len(change))]
                       ]
            except Exception as e:
                top = [
                    (a, change[a])
                    for a in change
                       ]
                print(e)

            # Select one of the top plans randomly as a 'trial
            trials[r] = top[random.randrange(len(top))]
            # Update inverse mapping
            area = trials[r][0]
            try:
                if r not in inverse[area]:
                    inverse[area].append(r)
            except KeyError:
                inverse[area] = [r]

    return trials, inverse


def region_growth(args, seed=None):
    """ Regional growth procedure """

    # args = (popltn, capacity, adjacency, spas, spas_nbr, sch_spas_nbr, schlattr, regions, regids)
    spas, spas_nbr, sch_spas_nbr, schlattr = args[3], args[4], args[5], args[6]
    regions, regids = dict(), dict()
    random.seed(seed)

    # Enumerate regions and set status
    spas_list = [s for s in spas_nbr.keys()]
    for area in spas_list:
        regids[area] = -1  # -1 means not assigned to a region

    sch_list = []

    # Initialize the regions with seed (school-containing polygons)
    for area in sch_spas_nbr.keys():
        sch_code = spas[schlattr][area]

        regids[area] = sch_code   # Assign the SCH CODE to the area
        regions[sch_code] = get_params(STATE=[area],
                                       SCH=sch_code)
        spas_list.remove(area)
        sch_list.append(sch_code)

    update_property(sch_list, args, regions)

    trials, inverse = find_trials(sch_list,
                                  {},
                                  {},
                                  regions, regids, args)

    # While there remains unassigned areas continue
    while spas_list:

        # Find the best trial plan
        best_val, area, regid = float("inf"), None, None

        for r in trials:
            a, val = trials[r]
            if val < best_val:
                best_val = val
                regid = r
                area = a

        if area:
            '''Assign area <a> to region <r>'''
            # print(area, regids[area])
            try:
                spas_list.remove(area)
            except Exception as e:
                print(e)
                print('Error caused by {} area: {} !!'
                      ' \n Terminating!!'.format(area,
                                                 ('already assigned',
                                                  'unassigned')[regids[area] == -1])
                      )

            # Make updates in the partition
            regids[area] = regid
            regions[regid]['STATE'].append(area)
            update_property([regid], args, regions)

            # Update the trial solutions for
            modify = [r for r in inverse[area]]

            if modify:
                # Regions containing 'area' as their top solution
                for r in modify:
                    del trials[r]

                del inverse[area]
                trials, inverse = find_trials(modify,
                                              trials,
                                              inverse,
                                              regions, regids, args)

    if list(regids.values()).count(-1) > 0:
        print("Error! Unassigned areas left. \n Terminating!!")
        exit(0)

    return regions, regids


def exist_init(args):
    """Extracts the existing partition for evaluation"""

    regions = dict()
    regids = dict()

    # args = (popltn, capacity, adjacency, spas, spas_nbr, sch_spas_nbr, schlattr, regions, regids)
    # Enumerate regions and set status
    spas, spas_nbr, schlattr = args[3], args[4], args[6]
    spas_list = [s for s in spas_nbr.keys()]

    for area in spas_list:
        regids[area] = -1  # -1 means not assigned to a region

    # Get the existing partition from the data
    for index, spa in spas.iterrows():
        area = index  # spa['SPA']
        regid = spa[schlattr]
        regids[area] = regid

        if regid not in regions.keys():
            regions[regid] = get_params(STATE=[], SCH=regid)

        regions[regid]['STATE'].append(area)

    return regions, regids


def infeasible(args, seed=None):
    """The resultant regions maintain contiguity constraint but might not contain one school per partition"""

    random.seed(seed)

    regions = dict()
    regids = dict()

    # args = (popltn, capacity, adjacency, spas, spas_nbr, sch_spas_nbr, schlattr, regions, regids)
    '''Enumerate regions and set status'''
    spas_nbr = args[4]
    spas_list = [x for x in spas_nbr.keys()]

    for area in spas_list:
        regids[area] = -1  # -1 means not assigned to a region

    exist_regions = exist_init(args)[0]  # get existing partition

    # Initialize the M regions with M areas (keeping some familiarity with existing partition)
    for regid in exist_regions.keys():

        exist_region = [m for m in exist_regions[regid]['STATE']]
        area = exist_region[random.randrange(len(exist_region))]  # pick random area from region

        regions[regid] = dict()
        regions[regid]['STATE'] = [area]
        regids[area] = regid  # key is the 'school_name'

        spas_list.remove(area)

    sch_list = [s for s in exist_regions.keys()]
    num_regions = len(sch_list)

    # Put the unassigned polygons in the regions
    while len(spas_list) > 0:
        regid = sch_list[random.randrange(num_regions)]  # pick randomly a region to grow
        region = [m for m in regions[regid]['STATE']]
        neighbors = get_adj_areas(region, spas_nbr, regids)

        if len(neighbors) > 0:

            index = random.randrange(len(neighbors))
            area = neighbors[index]

            regions[regid]['STATE'].append(area)
            regids[area] = regid

            spas_list.remove(area)

    return regions, regids


def initialize(option, args, seed=None):
    """Initializes regions with different schemes"""

    if option == 1:
        initialization = 'seeded'
        regions, regids = seeded_init(args, seed)

    elif option == 2:
        initialization = 'regional'
        regions, regids = region_growth(args, seed)

    elif option == 3:
        initialization = 'existing'
        regions, regids = exist_init(args)

    else:
        initialization = 'infeasible'
        regions, regids = infeasible(args, seed)

    return regions, regids, initialization


def par_initialize(i, arg, seed=None):

    regions, regids, init_type = initialize(i, arg, seed)
    update_property(regions.keys(), arg, regions)

    return regions, regids


def gen_solutions(arg=None, i=1, runs=0, seeds=None):
    """Generate solutions using the initialize()"""
    solutions = dict()

    try:

        # Parallel solution initialization
        if runs > 0:
            t = time.time()
            pool = mp.Pool(processes=min(5, runs))
            output = [(r, pool.apply_async(par_initialize,
                                           args=(i,
                                                 arg,
                                                 (None, seeds[r])[seeds is not None]
                                                 )
                                           )
                       )
                      for r in range(runs)
                      ]
            pool.close()

            for r, p in output:
                regions, regids = p.get()
                solutions[r] = get_partition(regions,
                                             regids)

            print('\n\n Done.. ')
            telapsed = time.time() - t
            print('\n\n Time taken: {:.4} min\n'.format(telapsed/60.0))

    except Exception as e:
        print(e)
        print("Couldn\'t generate solutions!!")

    return solutions
