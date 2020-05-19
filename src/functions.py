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


def obj_func(ids, zones):
    """Sum of the F-values"""

    F = sum(zones[r]['F'] for r in ids)
    return F


def show_stat(args):
    # args = (population, capacity, adjacency, spas, spas_nbr, sch_spas_nbr, sch, zones, zoneids)
    zones = args[7]

    for zoneid in zones.keys():

        # Get population and capacity statistics
        cap = zones[zoneid]['Capacity']
        pop = zones[zoneid]['Population']

        # Get perimeter and area statistics
        area = zones[zoneid]['Area']
        peri = zones[zoneid]['Perimeter']

        print("%s : Area: %10.5f Perimeter: %10.5f Capacity: %5d Population: %5d" %(zoneid, area, peri, cap, pop))


def find_change(zoneids, area, args):
    """Compute the change in objective function for adding 'area' to zone 'zoneid'."""
    # args = (population, capacity, adjacency, spas, spas_nbr, sch_spas_nbr, schlattr, zones, zoneids)

    zones = args[7]
    donor = zoneids[0]
    recipient = zoneids[1]

    donor_zone = [x for x in zones[donor]['STATE']]
    recip_zone = [x for x in zones[recipient]['STATE']]

    len_donor, len_recip = len(donor_zone), len(recip_zone)
    change, possible = None, False
    try:
        donor_zone.remove(area)
        recip_zone.append(area)
        new_zones = [donor_zone, recip_zone]

        # Compute the change in functional value
        initial = sum([zones[r]['F'] for r in zoneids])

        possible = True
        global w1, w2
        final = 0

        for s in new_zones:
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
    """Balance of population with school capacity of the zone"""
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
    """ Get perimeter, area and the target compactness of the zone"""
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


def computation(zoneid, args, zones=None):
    # args = (population, capacity, adjacency, spas, spas_nbr, sch_spas_nbr, schlattr, zones, zoneids)
    d = dict()

    if zones is None:
        zones = args[7]

    members = [m for m in zones[zoneid]['STATE']]

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

    return zoneid, d


def update_property(ids, args, zones=None):
    """Update the properties of zones (clusters) contained in zoneid_list"""

    # args = (population, capacity, adjacency, spas, spas_nbr, sch_spas_nbr, schlattr, zones, zoneids)
    if zones is None:
        zones = args[7]

    for r in ids:

        t, d = computation(r, args, zones)
        zones[r]['Capacity'] = d['Capacity']
        zones[r]['Population'] = d['Population']

        zones[r]['Area'] = d['Area']
        zones[r]['Perimeter'] = d['Perimeter']

        zones[r]['F1'] = d['F1']
        zones[r]['F2'] = d['F2']
        zones[r]['F'] = d['F']
