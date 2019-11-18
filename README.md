# REGAL

## Description
This is the repository for the SIGSPATIAL 2019 paper REGAL: A Regionalization framework for school boundaries

## Installation

The code is written in Python3 and the experiments were run on a machine using Ubuntu 18.04.3 LTS. You can follow the commands below for setting up your project.

### Setting up virtual environment
Assuming you have Python3, set up a virtual environment
```
pip install virtualenv
virtualenv -p /usr/bin/python3 ../venv
```

### Activate the environment
Always make sure to activate it before running any python script of this project
```
source ../venv/bin/activate
```

## Package installation
Install the required packages contained in the file packages.txt. This is a one-time thing, make sure the virtual environment is activated before performing this step.
```
pip install -r packages.txt
```

### Deactivate the environment
Deactivate it before exiting the project
```
deactivate
```

## Run the code
Code to simulate runs for district A.
1. Elementary school
```
./run_algo.py -s ES -d A
```
2. Middle school
```
./run_algo.py -s MS -d A
```
3. High school
```
./run_algo.py -s HS -d A
```
Or you can run all the experiments using the Makefile as
```
make district_A
```
You can similarly do it for district B.

## Cite
Please cite our paper if you use this code for your work:
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
