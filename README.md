# REGAL

## Description
Code repository for the SIGSPATIAL 2019 paper [REGAL: A Regionalization framework for school boundaries](http://doi.acm.org/10.1145/3347146.3359377)

## Installation

The code is written in Python3.6 and the experiments were run on a machine using Ubuntu 18.04.3 LTS. You can follow the commands below for setting up your project.

### Setting up virtual environment
Assuming you have Python3, set up a virtual environment
```
pip install virtualenv
virtualenv -p /usr/bin/python3 ./venv
```

### Activate the environment
Always make sure to activate it before running any python script of this project
```
source ./venv/bin/activate
```

## Package installation
Install the required packages contained in the file requirements.txt. This is a one-time thing, make sure the virtual environment is activated before performing this step.
```
pip install -r requirements.txt
```

Note that some geospatial packages in Python require dependencies like [GDAL](https://gdal.org/) to be already installed.

### Navigate to the 'src' folder containing the source scripts
```
cd ./src
```

## Run the code
You can run the simulations with the following three local search techniques.
SHC: Stochastic Hill Climbing
 SA: Simulated Annealing
 TS: Tabu Search

You can run all the experiments using the following command:
```
make REGAL
```

OR


First, make the executable file system readable
'''
chmod u+x ./run_algo.py
'''

Then, simulate runs for
1. Elementary school
```
./run_algo.py -s ES
```
2. Middle school
```
./run_algo.py -s MS
```
3. High school
```
./run_algo.py -s HS
```


### Deactivate the environment
Deactivate it before exiting the project
```
deactivate
```


### Data
The geospatial data used here is of [LCPS](https://www.lcps.org/) school district for the school year 2019-20. The data has been pre-processed for usage and may not accurately represent the policies/figures of LCPS.

## Citation
If you use this data/code for your work, please consider citing the paper:
```
@inproceedings{regal-gis,
 author = {Biswas, Subhodip and Chen, Fanglan and Chen, Zhiqian and Sistrunk, Andreea and Self, Nathan and Lu, Chang-Tien and Ramakrishnan, Naren},
 title = {REGAL: A Regionalization Framework for School Boundaries},
 booktitle = {Proceedings of the 27th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems},
 series = {SIGSPATIAL '19},
 year = {2019},
 isbn = {978-1-4503-6909-1},
 location = {Chicago, IL, USA},
 pages = {544--547},
 numpages = {4},
 url = {http://doi.acm.org/10.1145/3347146.3359377},
 doi = {10.1145/3347146.3359377},
 acmid = {3359377},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {constrained clustering, local search, optimization, spatial clustering},
} 
```
## Help
Should you have queries, feel free to send an email to subhodip [at] cs.vt.edu
