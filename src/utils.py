import os
import copy
import time
import random

import multiprocessing as mp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from functions import obj_func,\
    update_property,\
    find_change,\
    parameters,\
    target_compact, target_balance


class Utils:
    """
    Utility class containing necessary functions..
    """

    def __init__(self, args=None, seeds=None):
        self.args = args
        self.seeds = seeds

    def gen_solutions(self, init=1, num_sol=1, args=None, seeds=None):
        """Generate solutions using the initialize()"""

        if args is None:
            args = self.args
        if seeds is None:
            seeds = self.seeds

        solutions = dict()

        try:
            if num_sol > 0:
                t = time.time()
                if 0 < init < 3:
                    # Parallel (asynchronous) solution initialization
                    pool = mp.Pool(processes=min(mp.cpu_count() - 1, num_sol))
                    output = [(i,
                               pool.apply_async(self.initialize,
                                                args=(init,
                                                      args,
                                                      (None, seeds[i])[seeds is not None]
                                                      )
                                                )
                               )
                              for i in range(num_sol)
                              ]
                    pool.close()

                    for i, p in output:
                        zones, zoneids = p.get()
                        solutions[i] = self.get_partition(zones, zoneids)
                        
                else:
                    # Serial execution applies for 'existing' solution
                    zones, zoneids = self.initialize(init, args)
                    solutions = self.get_partition(zones, zoneids)
                    
                print('\n Done.. ')
                telapsed = time.time() - t
                print('\n Time taken: {:.4} min\n'.format(telapsed/60.0))

        except Exception as e:
            print(e)
            print("Couldn\'t generate solution(s)!!")

        return solutions

    def initialize(self, init, args=None, seed=None):
        """Initializes zones with different schemes"""

        if args is None:
            args = self.args

        # 'seeded' initialization
        if init == 1:
            zones, zoneids = self.seeded_init(args, seed)

        # 'infeasible' initialization that doesn't satisfy contiguity of zones
        elif init == 2:
            zones, zoneids = self.infeasible(args, seed)

        # 'existing' initialization
        elif init == 3:
            zones, zoneids = self.exist_init(args)

        else:
            pass

        # This is call to an outside method, needs to be resolved
        update_property(zones.keys(), args, zones)

        return zones, zoneids

    def seeded_init(self, args=None, seed=None):
        """ Seeded initialization starting with school containing polygons """

        random.seed(seed)
        if args is None:
            args = self.args
        # args = (population, capacity, adjacency, spas, spas_nbr, sch_spas_nbr, schlattr, zones, zoneids)

        zones, zoneids = dict(), dict()
        spas, spas_nbr, sch_spas_nbr, schlattr = args[3], args[4], args[5], args[6]

        # Enumerate zones and set status
        spas_list = [s for s in spas_nbr.keys()]
        for area in spas_list:
            zoneids[area] = -1  # -1 means not assigned to a zones

        # Initialize the zones with school-containing polygons
        sch_list = []
        for area in sch_spas_nbr.keys():
            sch_code = spas[schlattr][area]
            zoneids[area] = sch_code  # Assign the SCH_CODE to the area
            spas_list.remove(area)      # Remove areas that have already been assigned to a zone
            sch_list.append(sch_code)
            zones[sch_code] = self.get_params(STATE=[area], SCH=sch_code)

        num_zones = len(sch_list)     # No. of schools

        while len(spas_list) > 0:
            # Pick a random zones
            zoneid = sch_list[random.randrange(num_zones)]
            members = [x for x in zones[zoneid]['STATE']]
            neighbors = self.get_adj_areas(members, spas_nbr, zoneids)   # Get list of free areas around it

            if len(neighbors) > 0:
                area = neighbors[random.randrange(len(neighbors))]
                zoneids[area] = zoneid
                zones[zoneid]['STATE'].append(area)
                spas_list.remove(area)

        if list(zoneids.values()).count(-1) > 0:
            print('There are unassigned polygons present. Error!!')

        return zones, zoneids

    def exist_init(self, args=None):
        """
        Extracts the existing partition for evaluation
        """
        if args is None:
            args = self.args

        zones, zoneids = dict(), dict()
        # args = (population, capacity, adjacency, spas, spas_nbr, sch_spas_nbr, schlattr, zones, zoneids)

        # Enumerate zones and set status
        spas, spas_nbr, schlattr = args[3], args[4], args[6]
        spas_list = [s for s in spas_nbr.keys()]

        for area in spas_list:
            zoneids[area] = -1  # -1 means not assigned to a zones

        # Get the existing partition from the data
        for index, spa in spas.iterrows():
            area = spa['SPA']
            zoneid = spa[schlattr]
            zoneids[area] = zoneid

            if zoneid not in zones.keys():
                zones[zoneid] = self.get_params(STATE=[], SCH=zoneid)

            zones[zoneid]['STATE'].append(area)

        return zones, zoneids

    def infeasible(self, args=None, seed=None):
        """
        The resultant zones maintain contiguity constraint but might not contain one school per partition.
        """
        if args is None:
            args = self.args

        zones, zoneids = dict(), dict()
        random.seed(seed)

        # args = (population, capacity, adjacency, spas, spas_nbr, sch_spas_nbr, schlattr, zones, zoneids)
        # Enumerate zones and set status
        spas_nbr = args[4]
        spas_list = [x for x in spas_nbr.keys()]

        for area in spas_list:
            zoneids[area] = -1  # -1 means not assigned to a zones

        exist_zones, _ = self.exist_init()  # get existing partition

        # Initialize the M zones with M areas (keeping some familiarity with existing partition)
        for zoneid in exist_zones.keys():
            exist_zones = [m for m in exist_zones[zoneid]['STATE']]
            area = exist_zones[random.randrange(len(exist_zones))]  # pick random area from zones

            zones[zoneid] = dict()
            zones[zoneid]['STATE'] = [area]
            zoneids[area] = zoneid  # key is the 'school_name'
            spas_list.remove(area)

        sch_list = [s for s in exist_zones.keys()]
        num_zones = len(sch_list)

        # Put the unassigned polygons in the zones
        while len(spas_list) > 0:
            zoneid = sch_list[random.randrange(num_zones)]  # pick randomly a zones to grow
            members = [m for m in zones[zoneid]['STATE']]
            neighbors = self.get_adj_areas(members, spas_nbr, zoneids)

            if len(neighbors) > 0:

                index = random.randrange(len(neighbors))
                area = neighbors[index]
                zones[zoneid]['STATE'].append(area)
                zoneids[area] = zoneid
                spas_list.remove(area)

        return zones, zoneids

    @staticmethod
    def get_neighbors(zoneid, args):
        """Get the list of areas adjacent to the base zones"""

        # args = (population, capacity, adjacency, spas, spas_nbr, sch_spas_nbr, schlattr, zones, zoneids)
        sch_spas_nbr = args[5]

        sch_spas = [x for x in sch_spas_nbr.keys()]
        spas_nbr = args[4]
        zones = args[7]
        zoneids = args[8]

        neighbors = []
        for area in zones[zoneid]['STATE']:
            neighbors = neighbors + [x for x in spas_nbr[str(area)]
                                     if zoneids[x] != zoneid and x not in sch_spas]

        neighbors = list(set(neighbors))  # get unique values

        # Check which areas break contiguity on being swapped
        to_remove = []

        for area in neighbors:

            donor_zoneid = zoneids[area]
            donor_zones = [m for m in zones[donor_zoneid]['STATE']]

            if area not in donor_zones:
                pass
                # print(donor_zoneid, ':', donor_cluster)
            else:
                donor_zones.remove(area)

            if len(donor_zones) > 1:  # If the cluster is not a singleton

                adjacency = args[2]
                donor_zones_adj = adjacency.loc[donor_zones, donor_zones].values
                adjacent_mat = csr_matrix(donor_zones_adj)
                num_connect_comp = connected_components(adjacent_mat, directed=False, return_labels=False)

                if num_connect_comp != 1:  # Not 1 means disconnected
                    to_remove.append(area)

        # Remove those areas that break contiguity
        for area in to_remove:
            neighbors.remove(area)

        return neighbors

    @staticmethod
    def get_adj_areas(areas, spas_nbr, zoneids):
        """Returns adjacent unassigned area polygons to a cluster"""
        adj_areas = []
        if len(areas) > 0:
            for area in areas:
                adj_areas = adj_areas + [a for a in spas_nbr[area]
                                         if zoneids[a] == -1]

            adj_areas = list(set(adj_areas))

        return adj_areas

    def get_partition(self, zones, zoneids):
        """

        """
        r = copy.deepcopy(zones)
        i = copy.deepcopy(zoneids)
        f = obj_func(r.keys(), r)

        partition = self.get_params(zones=r,
                                    zoneIds=i,
                                    FuncVal=f)
        return partition

    @staticmethod
    def make_move(cids, area, args):
        """Moving polygon between clusters"""

        donor = cids[0]
        recip = cids[1]
        # args = (population, capacity, adjacency, spas, spas_nbr, sch_spas_nbr, sch, zones, zoneids)
        zones = args[7]

        donor_zones = [m for m in zones[donor]['STATE']]
        recip_zones = [m for m in zones[recip]['STATE']]

        moved = False
        try:
            if len(donor_zones) > 1:
                # Make the move
                donor_zones.remove(area)
                recip_zones.append(area)

                # Update the zones
                zones[donor]['STATE'] = donor_zones
                zones[recip]['STATE'] = recip_zones

                zoneids = args[8]
                zoneids[area] = recip

                update_property(cids, args)
                moved = True

        except Exception as e:
            print(e)
            print('Exception in make_move')

        return moved

    def get_alg_params(self, run, args, *argv):
        # argv = (iteration, t_elapsed, termination, seed, initialization, sch, inital, final)
        iteration = argv[0]
        t_elapsed = argv[1]
        terminate = argv[2]
        seed = argv[3]
        initialize = argv[4]
        sch = argv[5]

        """Consolidating all the attributes"""
        # Constants defined as global variables
        w1, w2, epsilon, max_iter = parameters()

        alg_params = self.get_params(w1=w1,
                                     w2=w2,
                                     epsilon=epsilon,
                                     MaxIter=max_iter)

        params = self.get_params(AlgParams=alg_params,
                                 Iteration=iteration,
                                 TimeElapsed=t_elapsed,
                                 Termination=terminate,
                                 Seed=seed,
                                 School=sch,
                                 Initialization=initialize)

        initial, final = argv[6], argv[7]
        existing = self.gen_solutions(3)

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

    @staticmethod
    def create_dir(*args):
        """
        Function to create directories and generate paths.
        """
        write_path = "../"
        for name in args:
            try:
                write_path =  write_path + "{}/".format(name)
                os.mkdir(write_path)
            except Exception as e:
                print(".")

        return write_path

    @staticmethod
    def get_params(**argsv):
        # Returns a dictionary
        return argsv

    @staticmethod
    def get_args(*args):
        # Returns a tuple
        return args
