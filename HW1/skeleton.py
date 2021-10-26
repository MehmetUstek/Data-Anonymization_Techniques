##############################################################################
# This skeleton was created by Mandana Bagheri (mmarzijarani20@ku.edu.tr)    #
# Note: requires Python 3.5+                                                 #
##############################################################################

import csv
import glob
import os
import sys
from treelib import Node, Tree
from collections import Counter

from copy import deepcopy
from typing import Optional

if sys.version_info[0] < 3 or sys.version_info[1] < 5:
    sys.stdout.write("Requires Python 3.x.\n")
    sys.exit(1)

##############################################################################
# Helper Functions                                                           #
# These functions are provided to you as starting points. They may help your #
# code remain structured and organized. But you are not required to use      #
# them. You can modify them or implement your own helper functions.          #
##############################################################################

def read_dataset(dataset_file: str):
    """ Read a dataset into a list and return.

    Args:
        dataset_file (str): path to the dataset file.

    Returns:
        list[dict]: a list of dataset rows.
    """
    result = []
    with open(dataset_file) as f:
        records = csv.DictReader(f)
        for row in records:
            result.append(row)
    # print(result[0]['age']) # debug: testing.
    return result


def write_dataset(dataset, dataset_file: str) -> bool:
    """ Writes a dataset to a csv file.

    Args:
        dataset: the data in list[dict] format
        dataset_file: str, the path to the csv file

    Returns:
        bool: True if succeeds.
    """
    assert len(dataset)>0, "The anonymized dataset is empty."
    keys = dataset[0].keys()
    with open(dataset_file, 'w', newline='')  as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(dataset)
    return True

# class Tree:
#     def __init__(self, val, queue, indentation):
#         self.val = val
#         self.children = []
#
#     def add_children(self,val, queue, indentation):
#         child_node = Tree(val, queue, indentation)
#         self.children.append(child_node)
#         return child_node
#     # def get_children(self, queue, indentation):
#     #     for c in self.children:
#     #         if self.children[c] ==
#     def __repr__(self):
#         return f"Tree({self.val}): {self.children}"
#     # def get(self, root, indentation, queue):
#     #     i = 0
#     #     q = 0
#     #     while q!=queue:
#     #         q += 1
#     #         root =
#     #         while i!=indentation:
#     #             i += 1
#
#     def tree_root(self):
#         return

def read_DGH(DGH_file: str):
    """ Reads one DGH file and returns in desired format.

    Args:
        DGH_file (str): the path to DGH file.
    """
    # Returned in a tree structure

    result = []
    with open(DGH_file) as f:
        records = csv.reader(f, delimiter="\t")
        for row in records:
            result.append(row)

    tree = Tree()
    total_children = 1
    parent = tree.create_node(result[0][0],"root", data=0)
    root = parent

    i = 0
    total_children += 1
    indentation_level = 0
    for c in result:
        if len(c) ==1:
            pass
        else:
            indentation_level = len(c)
            # print(indentation_level)
        # Top down
        if i!= 0:
            if parent.is_root():
                parent = tree.create_node(result[i][1], parent=root, data=indentation_level)
            else:
                if indentation_level > parent.data:
                    parent = tree.create_node(result[i][indentation_level-1], parent=parent, data=indentation_level)
                else:
                    if parent.data:
                        # Bottom up
                        while indentation_level <= parent.data:
                            parent = tree.parent(parent.identifier)
                        parent = tree.create_node(result[i][indentation_level-1], parent=parent,data = indentation_level)

        i += 1


    # tree.show()
    result = tree
    return result


def read_DGHs(DGH_folder: str) -> dict:
    """ Read all DGH files from a directory and put them into a dictionary.

    Args:
        DGH_folder (str): the path to the directory containing DGH files.

    Returns:
        dict: a dictionary where each key is attribute name and values
            are DGHs in your desired format.
    """
    DGHs = {}
    for DGH_file in glob.glob(DGH_folder + "/*.txt"):
        attribute_name = os.path.basename(DGH_file)[:-4]
        DGHs[attribute_name] = read_DGH(DGH_file);

    return DGHs


##############################################################################
# Mandatory Functions                                                        #
# You need to complete these functions without changing their parameters.    #
##############################################################################


def cost_MD(raw_dataset_file: str, anonymized_dataset_file: str,
    DGH_folder: str) -> float:
    """Calculate Distortion Metric (MD) cost between two datasets.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        anonymized_dataset_file (str): the path to the anonymized dataset file.
        DGH_folder (str): the path to the DGH directory.

    Returns:
        float: the calculated cost.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    assert(len(raw_dataset)>0 and len(raw_dataset) == len(anonymized_dataset)
        and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)

    #TODO: complete this function.
    # print(DGHs)
    ctr_raw = Counter()
    for data in raw_dataset:
        for df, sa in data.items():
            ctr_raw[(df,sa)] += 1
    # print(ctr_raw)

    ctr_anon = Counter()
    for data in anonymized_dataset:
        for df, sa in data.items():
            ctr_anon[(df,sa)] += 1
    # print(ctr_anon)
    generalized = ctr_anon - ctr_raw
    lost = ctr_raw - ctr_anon
    print("generalized:\n", generalized)
    print("lost:\n", lost)

    # for j, k in generalized.keys():
    #     print("j", j)
    #     print("k", k)
    #     print(lost.keys())
    list = []
    for key in generalized.keys():
        list.append(key)

    print(list)
    for (i,j),k in lost.items():
        print("i",i)
        print("j",j)
        print("k",k)
        # if list[i] == 'gender':
        #     print(1)
        if generalized[(i,)]:
            print(1)

        # print(generalized.keys())


        for dgh,tree in DGHs.items():
            pass
            # print(dgh)
            # print(tree)


    return -1


def cost_LM(raw_dataset_file: str, anonymized_dataset_file: str,
    DGH_folder: str) -> float:
    """Calculate Loss Metric (LM) cost between two datasets.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        anonymized_dataset_file (str): the path to the anonymized dataset file.
        DGH_folder (str): the path to the DGH directory.

    Returns:
        float: the calculated cost.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    assert(len(raw_dataset)>0 and len(raw_dataset) == len(anonymized_dataset)
        and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)

    #TODO: complete this function.
    return -1


def random_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
    output_file: str):
    """ K-anonymization a dataset, given a set of DGHs and a k-anonymity param.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)

    anonymized_dataset = []
    #TODO: complete this function.

    write_dataset(anonymized_dataset, output_file)


def clustering_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
    output_file: str):
    """ Clustering-based anonymization a dataset, given a set of DGHs.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)

    anonymized_dataset = []
    #TODO: complete this function.

    write_dataset(anonymized_dataset, output_file)


def topdown_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
    output_file: str):
    """ Top-down anonymization a dataset, given a set of DGHs.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)

    anonymized_dataset = []
    #TODO: complete this function.

    write_dataset(anonymized_dataset, output_file)

# print(read_DGHs("DGHs"))
cost_MD("adult-hw1.csv","adult-anonymized.csv", "DGHs" )

# Command line argument handling and calling of respective anonymizer:
if len(sys.argv) < 6:
    print(f"Usage: python3 {sys.argv[0]} algorithm DGH-folder raw-dataset.csv anonymized.csv k")
    print(f"\tWhere algorithm is one of [clustering, random, topdown]")
    sys.exit(1)

algorithm = sys.argv[1]
if algorithm not in ['clustering', 'random', 'topdown']:
    print("Invalid algorithm.")
    sys.exit(2)

dgh_path = sys.argv[2]
raw_file = sys.argv[3]
anonymized_file = sys.argv[4]
k = int(sys.argv[5])

function = eval(f"{algorithm}_anonymizer");
function(raw_file, dgh_path, k, anonymized_file)

cost_md = cost_MD(raw_file, anonymized_file, dgh_path)
cost_lm = cost_LM(raw_file, anonymized_file, dgh_path)
print (f"Results of {k}-anonimity:\n\tCost_MD: {cost_md}\n\tCost_LM: {cost_lm}\n")


# Sample usage:
# python3 code.py clustering DGHs/ adult-hw1.csv result.csv 300