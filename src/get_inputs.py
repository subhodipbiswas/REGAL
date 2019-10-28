import os
import json
import geopandas as gpd
from os.path import join
import pandas as pd


def get_params(**argsv):
    # Returns a dictionary
    return argsv


def get_args(*args):
    # Returns a tuple
    return args


def get_adj_list(spas_nbrlist):
    """Returns adjacency relations as a data frame"""
    spas_list = [x for x in spas_nbrlist.keys()]
    adjacency = pd.DataFrame(0, index=spas_list, columns=spas_list)

    for spa in spas_list:
        for nbr in spas_nbrlist[str(spa)]:
            adjacency.at[spa, nbr] = 1

    return adjacency


def get_schattr(sch, district):
    """ Get school attribute used later """
    if district == 'A':
        # District A
        if sch == 'ES':
            sch_attr = "ES_ID"
        elif sch == 'MS':
            sch_attr = "MS_ID"
        elif sch == 'HS':
            sch_attr = "HS_ID"
            
    elif district == 'B':
        # District B
        if sch == 'ES':
            sch_attr = "ELEM_CODE"
        elif sch == 'MS':
            sch_attr = "MID_CODE"
        elif sch == 'HS':
            sch_attr = "HIGH_CODE"
        
    return sch_attr


def create_dir(*args):
    """Function to create directories and generate paths"""
    write_path = "../"
    for name in args:
        try:
            write_path =  write_path +"{}/".format(name)
            os.mkdir(write_path)
            print(name)
        except Exception as e:
            print(".")
    
    # print("Saving results to {}".format(write_path))

    return write_path


def enforce_const(spas_nbrlist, sch_spas_nbrlist):
    """Remove links between polygons that contain school inside them"""
    sch_spas_list = list(sch_spas_nbrlist.keys())

    for spa in sch_spas_list:

        if len(sch_spas_nbrlist[spa]) > 0:
            ref_list = sch_spas_nbrlist[spa]
            # print(spa, ref_list)
            for item in ref_list:
                # Adjacency is a symmetric relation
                spas_nbrlist[spa].remove(item)


def modify_spas(spas, district):
    """Modify the indices of the SPA dataframes"""

    if district == 'A':
        indices = list(spas['SPA'])
        index_list = [str(x) for x in indices]
        spas.index = index_list

    elif district == 'B':
        spas.index = list(spas['STDYAREA'])  # STDYAREA is same as SPA
    return spas


def read_json_file(file_loc):
    """ Reads a json file given the file location """
    with open(file_loc) as json_data:
        nbr_list = json.load(json_data)
        json_data.close()

    return nbr_list


def get_data_a(sch, sy, spas, schools):
    """  Get attending student population and capacity for schools in District A """
    popltn = dict()
    capacity = dict()

    if sch == "ES":
        proprty = 'Elementary'
    elif sch == "MS":
        proprty = 'Middle'
    elif sch == "HS":
        proprty = 'High'

    for index, spa in spas.iterrows():
        area_name = index  # str(spa['SPA'])
        popltn[area_name] = spa[proprty]
        capacity[area_name] = 0

    proprty = 'DESGN_CAP_' + sy    # change
    for index, school in schools.iterrows():
        area_name = str(school['SPA'])
        capacity[area_name] = school[proprty]

    return popltn, capacity


def get_data_b(sch, sy, spas, schools):
    """  Get attending student population and capacity for schools in District B """
    popltn = dict()
    capacity = dict()

    # determine the school population of each school
    for index, spa in spas.iterrows():
        area_name = spa['STDYAREA']

        if sch == "ES":
            popltn[area_name] = spa['TOTAL_K_5'] + spa['PK'] + spa['Other']
        elif sch == "MS":
            popltn[area_name] = spa['TOTAL_6_8']
        elif sch == "HS":
            popltn[area_name] = spa['TOTAL_9_12']

        capacity[area_name] = 0

    if sch == "ES":
        proprty = 'ELEMENTARY'
    elif sch == "MS":
        proprty = 'MIDDLE'
    elif sch == "HS":
        proprty = 'HIGH'

    # only spas containing school will have school capacity
    for index, school in schools.iterrows():
        area_name = school['STDYAREA']
        sch_type = school['CLASS']

        if sch_type == proprty:
            capacity[area_name] = school['CAPACITY']

    return popltn, capacity


def update_list(spas_nbrlist, sch_spas_nbrlist):
    """Convert data types into strings"""

    spas_list = [i for i in spas_nbrlist.keys()]
    for key in spas_list:
        temp_list = [str(i) for i in spas_nbrlist[key]]
        spas_nbrlist[key] = temp_list

    sch_spas_list = [i for i in sch_spas_nbrlist.keys()]
    for key in sch_spas_list:
        temp_list = [str(i) for i in sch_spas_nbrlist[key]]
        sch_spas_nbrlist[key] = temp_list


def get_inputs(*argv):
    """ Read geospatial data corresponding to school districts """

    # argv = (sch, sy, read_path, district)
    sch = argv[0]
    sy  = argv[1]
    read_path = argv[2]
    district  = argv[3]

    spas_nbrlist = read_json_file(join(read_path,
                                       "SPA",
                                       "neighbors",
                                       "nbrlist_SPA_SY" + sy + ".json")
                                  )
    # Read data for district A
    if district == 'A':
        # Read the shapefiles for SPAs
        spas = modify_spas(gpd.read_file(join(read_path,
                                              "SPA",
                                              "final",
                                              "SPA_SY" + sy + ".json")
                                         ),
                           district)
        # Read adjacency relationship of the SPAs
        sch_spas_nbrlist = read_json_file(join(read_path,
                                               "Schools_SY2017_18",
                                               "neighbors",
                                               "nbrlist_" + sch + "contain_SPA.json")
                                          )
        # Read the shapefiles corresponding to the schools
        schools = gpd.read_file(join(read_path,
                                     "Enrollment",
                                     sch + "_Info.json")
                                )
        # Read the population and capacity info the schools
        popltn, capacity = get_data_a(sch, sy, spas, schools)

    # Read data for school B
    elif district == 'B':
        spas = modify_spas(gpd.read_file(join(read_path,
                                              "SPA",
                                              "SPA_SY" + sy + ".json")
                                         ),
                           district)
        sch_spas_nbrlist = read_json_file(join(read_path,
                                               "Schools",
                                               "Schools_SY" + sy,
                                               "nbrlist_" + sch + "contain_SPA.json")
                                          )
        schools = gpd.read_file(join(read_path,
                                     "Schools",
                                     "Schools_SY" + sy + ".json")
                                )
        popltn, capacity = get_data_b(sch, sy, spas, schools)

    update_list(spas_nbrlist, sch_spas_nbrlist)
    enforce_const(spas_nbrlist, sch_spas_nbrlist)  # remove adjacency links between school containing polygons
    adjacency = get_adj_list(spas_nbrlist)     # get the adjacency matrix of the graph

    return spas, spas_nbrlist, sch_spas_nbrlist, adjacency, popltn, capacity
