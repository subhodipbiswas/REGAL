# REGAL

## Description
This is the repository for the SIGSPATIAL 2019 paper REGAL: A Regionalization framework for school boundaries

## Installation

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
@inproceedings{regal_gis19,
    title = {REGAL: A Regionalization framework for school boundaries},
    author = {Subhodip Biswas and Fanglan Chen and Zhiqian Chen and Andreea Sistrunk and Nathan Self and Chang-Tien Lu and Naren Ramakrishnan},
    booktitle = {27th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems, (GIS)},
    year = {2019},
    url = {https://doi.org/10.1145/3347146.3359377}
}
```
## Help
Should you have queries, feel free to send an email to subhodip [at] cs.vt.edu
