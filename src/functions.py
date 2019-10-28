import math
from shapely.ops import unary_union

w1 = 0.0
w2 = 0.0
max_iter = 0
epsilon = pow(10,0)


def print_params():
    print(' w1 (Capacity) : {:.2}\n w2 (Compactness) : {:.2}'.format(w1, w2))
    print(' Epsilon : {}\n MaxIter : {}'.format(epsilon,
                                                max_iter
                                                )
          )


def set_params(w):
    global w1, w2, epsilon, max_iter
    w1 = 0.1*w
    w2 = 1-w1
    epsilon = pow(10, -5)  # global constant
    max_iter = 1000

    print_params()


def parameters():
    return w1, w2, epsilon, max_iter


def obj_func(ids, regions):
    """Sum of the F-values"""

    F = sum(regions[r]['F'] for r in ids)
    return F


def show_stat(args):
    # args = (popltn, capacity, adjacency, spas, spas_nbr, sch_spas_nbr, sch, regions, regids)
    regions = args[7]

    for regid in regions.keys():

        # Get population and capacity statistics
        cap = regions[regid]['Capacity']
        pop = regions[regid]['Population']

        # Get perimeter and area statistics
        area = regions[regid]['Area']
        peri = regions[regid]['Perimeter']

        print("%s : Area: %10.5f Perimeter: %10.5f Capacity: %5d Population: %5d" %(regid, area, peri, cap, pop))


def find_change(regids, area, args):
    """Compute the change in objective function for adding 'area' to region 'regid'."""
    # args = (popltn, capacity, adjacency, spas, spas_nbr, sch_spas_nbr, schlattr, regions, regids)

    regions = args[7]
    donor = regids[0]
    recipient = regids[1]

    donor_region = [x for x in regions[donor]['STATE']]
    recip_region = [x for x in regions[recipient]['STATE']]

    len_donor, len_recip = len(donor_region), len(recip_region)
    change, possible = None, False
    try:
        donor_region.remove(area)
        recip_region.append(area)
        new_regions = [donor_region, recip_region]

        # Compute the change in functional value
        initial = sum([regions[r]['F'] for r in regids])

        possible = True
        global w1, w2
        final = 0

        for s in new_regions:
            members = [m for m in s]

            f1 = target_balance(members, args[0], args[1])[2]
            f2 = target_compact(members, args[3])[2]
            final += w1 * f1 + w2 * f2

            if not members:
                possible = False

        change = final - initial

    except Exception as e:
        print(e)

    return change, possible


def target_balance(members, pop, cap):
    """Balance of population with school capacity of the region"""
    p = c = None
    try:
        p = sum([pop[m] for m in members])
        c = sum([cap[m] for m in members])
        score = (p + 0.001) / (c + epsilon)

    except Exception as e:
        print(e)
        score = 0

    f1 = abs(1 - score)
    return p, c, f1


def target_compact(members, shapes):
    """ Get perimeter, area and the target compactness of the region"""
    shape_list = [shapes['geometry'][m] for m in members]

    try:
        total = unary_union(shape_list)
        area = total.area
        peri = total.length
        score = (4*math.pi*area)/(peri**2)    # IPQ score or Polsby Popper score
        # score = peri**2/(4*math.pi*area)      # Schwartzberg's index
    except Exception as e:
        area, peri, score = 0, 0, 0

    f2 = 1 - score
    return area, peri, f2


def computation(regid, args, regions=None):
    # args = (popltn, capacity, adjacency, spas, spas_nbr, sch_spas_nbr, schlattr, regions, regids)
    d = dict()

    if regions is None:
        regions = args[7]

    members = [m for m in regions[regid]['STATE']]

    '''Get population and capacity statistics'''
    pop, cap, f1 = target_balance(members=members, pop=args[0], cap=args[1])
    d['Capacity'] = cap
    d['Population'] = pop
    d['F1'] = f1

    '''Get area and perimeter statistics'''
    area, peri, f2 = target_compact(members=members, shapes=args[3])
    d['Area'] = area
    d['Perimeter'] = peri
    d['F2'] = f2

    global w1, w2, epsilon
    try:
        F = w1 * f1 + w2 * f2
    except Exception as e:
        print(e)
        print('Error in computation()')
        F = 1

    d['F'] = F

    return regid, d


def update_property(ids, args, regions=None):
    """Update the properties of regions (clusters) contained in regid_list"""

    # args = (popltn, capacity, adjacency, spas, spas_nbr, sch_spas_nbr, schlattr, regions, regids)
    if regions is None:
        regions = args[7]

    for r in ids:

        t, d = computation(r, args, regions)
        regions[r]['Capacity'] = d['Capacity']
        regions[r]['Population'] = d['Population']

        regions[r]['Area'] = d['Area']
        regions[r]['Perimeter'] = d['Perimeter']

        regions[r]['F1'] = d['F1']
        regions[r]['F2'] = d['F2']
        regions[r]['F'] = d['F']
