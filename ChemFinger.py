import torch
import deepchem
from rdkit import Chem

SD_Name = r"c:\data\SOM_dataset.sdf"
SD_supp = Chem.SDMolSupplier(SD_Name)
mol_list = [ x for x in SD_supp]
prop_list = ['SOM1','SOM2','SOM3','SOM4','SOM5','SOM6']

from deepchem.models.layers import GraphConv, GraphPool, GraphGather
import tensorflow as tf



