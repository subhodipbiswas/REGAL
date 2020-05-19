import os
import json
import copy
import pandas as pd
import geopandas as gpd


class GetInputs:
    """
    Helps in reading the geospatial data files and returning the processed data.
    """
    sy = "SY2019_20"    # school year of the corresponding datasets
    
    def __init__(self, sch = 'ES'):
        self.sch = sch
        self.read_path = '../data/'

    def get_inputs(self):
        """ Read geo-spatial data corresponding to school districts """
        
        # Read the data: Schools and Student Planning Areas (SPAs)
        spas = gpd.read_file(self.read_path + "SPAs.json")
        schools = self.get_schools()

        schlattr = self.get_schlattr()

        # Get the adjacency list corresponding to the SPAs
        spas_nbrlist = self.read_json_file(self.read_path + "nbrlist_SPA.json")
        sch_spas_nbrlist = self.get_sch_spas(schools, spas_nbrlist)
        
        spas.index = list(spas['SPA'])
        
        # Read the population and capacity info the schools
        population, capacity = self.get_pop_cap(spas, schools)
        
        # Remove adjacency links between school containing polygons
        self.remove_links(spas_nbrlist, sch_spas_nbrlist)
        
        # Get the adjacency matrix of the underlying graph structure of the SPAs
        adjacency = self.get_adj_list(spas_nbrlist)

        return population, capacity, adjacency, spas, spas_nbrlist, sch_spas_nbrlist, schlattr

    def get_schools(self):
        """ Read the geospatial data corresponding to schools and filter them based on the school type. """
        schools = gpd.read_file(self.read_path + 'Schools.json')
        delete = list()
        
        for index, school in schools.iterrows():
            if school['SCHOOL_TYP'] != self.sch:
                delete.append(index)
                
        schools.drop(delete, axis=0, inplace=True)
        return schools

    def get_schlattr(self):
        """
        Get the attribute corresponding to the school level
        """
        if self.sch == 'ES':
            schlattr = 'ELEM_CODE'
        elif self.sch == 'MS':
            schlattr = 'MID_CODE'
        elif self.sch == 'HS':
            schlattr = 'HIGH_CODE'
        else:
            schlattr = None

        return schlattr

    def get_adj_list(self, spas_nbrlist):
        """Returns adjacency relations as a data frame"""
        spas_list = [x for x in spas_nbrlist.keys()]
        adjacency = pd.DataFrame(0, index=spas_list, columns=spas_list)

        for spa in spas_list:
            for nbr in spas_nbrlist[str(spa)]:
                adjacency.at[spa, nbr] = 1

        return adjacency

    def remove_links(self, spas_nbrlist, sch_spas_nbrlist):
        """
        Remove links between SPAs that contain school inside them. This artificial linkage removal helps in creating invalid school boundaries with two schools inside them.
        """
        
        sch_spas = list(sch_spas_nbrlist.keys())

        for spa in sch_spas:
            sch_neighbors = sch_spas_nbrlist[spa]
            
            if len(sch_neighbors) > 0:
                for n in sch_neighbors:
                    spas_nbrlist[spa].remove(n)

    @staticmethod
    def read_json_file(file_loc):
        """ Reads a json file given the file location """
        with open(file_loc) as inpfile:
            data = json.load(inpfile)
            inpfile.close()

        return data

    def get_sch_spas(self, schools, spas_nbrlist):
        """ Find the adjacency relations between SPAs containing schools """
        sch_spas_nbrlist = copy.deepcopy(spas_nbrlist)
        
        # Find the school containing SPAs
        sch_spas = list(schools['SPA'])
        
        # Remove keys that don't correspond to school containing SPAs
        delete = [s for s in sch_spas_nbrlist.keys() if s not in sch_spas]
                
        for spa in delete:
            del sch_spas_nbrlist[spa]
            
        # Update the neighbor list of the remaining SPAs
        for spa, neighbors in sch_spas_nbrlist.items():
            sch_spas_nbrlist[spa] = [n for n in neighbors if n in sch_spas]
            
        return sch_spas_nbrlist

    def get_pop_cap(self, spas, schools):
        """
        Get attending student population and capacity for schools in a school district.
        """
        population, capacity = dict(), dict()
        attribute = ''

        if self.sch == 'ES':
            attribute = 'ELEM_POP'
            
        elif self.sch == 'MS':
            attribute = 'MID_POP'
            
        elif self.sch == 'HS':
            attribute = 'HIGH_POP'

        try:
            for index, spa in spas.iterrows():
                spa_name = spa['SPA']
                capacity[spa_name] = 0
                population[spa_name] = spa[attribute]

            attribute = 'CAPACITY'    # Program Capacity
            for index, school in schools.iterrows():
                spa_name = school['SPA']
                capacity[spa_name] = school[attribute]
        except Exception as e:
            print(e)

        return population, capacity