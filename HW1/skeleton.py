##############################################################################
# This skeleton was created by Mandana Bagheri (mmarzijarani20@ku.edu.tr)    #
# Note: requires Python 3.5+                                                 #
##############################################################################

import csv
import glob
import os
import sys
import random
from _collections import defaultdict
from treelib import Node, Tree
from collections import Counter

from copy import deepcopy
from typing import Optional

if sys.version_info[0] < 3 or sys.version_info[1] < 5:
    sys.stdout.write("Requires Python 3.x.\n")
    sys.exit(1)

DGHs = {}
list_dgh = []
list_tree = []


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
    assert len(dataset) > 0, "The anonymized dataset is empty."
    keys = dataset[0].keys()
    with open(dataset_file, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(dataset)
    return True


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
    parent = tree.create_node(result[0][0].rstrip(), 'Any', data=1)
    root = parent
    i = 0
    total_children += 1
    indentation_level = 0
    for c in result:
        if len(c) == 1:
            pass
        else:
            c_count = 1
            for c_element in c:
                if c_element != '':
                    indentation_level = c_count
                    break
                c_count += 1

            # indentation_level = len(c)
            # print(indentation_level)
        # Top down
        if i != 0:
            if parent.is_root():
                parent = tree.create_node(result[i][1].rstrip(), identifier=result[i][1].rstrip(), parent=root,
                                          data=indentation_level)
            else:
                if indentation_level > parent.data:
                    parent = tree.create_node(result[i][indentation_level - 1].rstrip(),
                                              identifier=result[i][indentation_level - 1].rstrip(), parent=parent,
                                              data=indentation_level)
                else:
                    if parent.data:
                        # Bottom up
                        while indentation_level <= parent.data:
                            parent = tree.parent(parent.identifier)
                        parent = tree.create_node(result[i][indentation_level - 1].rstrip(),
                                                  identifier=result[i][indentation_level - 1].rstrip(), parent=parent,
                                                  data=indentation_level)

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
    # TODO: Really need an optimization, as this holds the whole function back by slowing it at least 10 times.
    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    global DGHs
    if not DGHs:
        assert (len(raw_dataset) > 0 and len(raw_dataset) == len(anonymized_dataset)
                and len(raw_dataset[0]) == len(anonymized_dataset[0]))
        DGHs = read_DGHs(DGH_folder)

    return MD_cost_of_a_table(raw_dataset, anonymized_dataset)


def MD_cost_of_a_table(eq1, eq2):
    # TODO: Gives wrong answers.
    global list_tree
    global list_dgh
    total_MD_cost = 0
    ctr_raw = Counter()
    for data in eq1:
        for df, sa in data.items():
            ctr_raw[(df, sa)] += 1

    ctr_anon = Counter()
    for data in eq2:
        for df, sa in data.items():
            ctr_anon[(df, sa)] += 1
    ctr_anon.subtract(ctr_raw)

    if not (list_dgh and list_tree):
        for dgh, tree in DGHs.items():
            list_dgh.append(dgh)
            list_tree.append(tree)
    tree1 = Tree()
    while ctr_anon:
        (i, j), k = ctr_anon.popitem()
        if k != 0:
            if not i in DGHs:
                continue
            index = list_dgh.index(i)
            tree1 = list_tree[index]
            current = tree1.get_node(j)
            depth1 = current.data

            nodes_list = [tree1[node].tag for node in tree1.expand_tree(mode=Tree.DEPTH)]
            for z in range(abs(k)):
                for n in nodes_list:
                    if (i, n) in ctr_anon:
                        node2 = tree1.get_node(n)
                        depth2 = node2.data
                        if k < 0:
                            cost = -(depth2 - depth1)
                        else:
                            cost = (depth2 - depth1)
                        if ctr_anon[(i, n)] > 0:
                            ctr_anon[(i, n)] -= 1
                        else:
                            ctr_anon[(i, n)] += 1

                        if ctr_anon[(i, n)] == 0:
                            del ctr_anon[(i, n)]
                        total_MD_cost += cost
                        break


    return abs(total_MD_cost)


def get_the_lowest_common_ancestor_for_two(tree, node1, node2):
    if node1 != node2:
        if node1.data > node2.data:
            while node1.data != node2.data:
                node1 = tree.parent(node1.identifier)
        elif node2.data > node1.data:
            while node1.data != node2.data:
                node2 = tree.parent(node2.identifier)
        if node1 == node2:
            return node1
        else:
            while node1 != node2:
                node1 = tree.parent(node1.identifier)
                node2 = tree.parent(node2.identifier)
            return node1
    return node1


def get_the_lowest_common_ancestor(tree, list_of_nodes: list):
    node = ''
    visited = []
    for i in range(len(list_of_nodes)):
        for j in range(len(list_of_nodes)):
            if i != j:
                if not (i, j) in visited:
                    i_node = list_of_nodes[i]
                    j_node = list_of_nodes[j]
                    node = get_the_lowest_common_ancestor_for_two(tree, i_node, j_node)
                    visited.append((i, j))
                    visited.append((j, i))
                else:
                    continue
        #     j_iteration += 1
        # i_iteration += 1
    return node


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
    assert (len(raw_dataset) > 0 and len(raw_dataset) == len(anonymized_dataset)
            and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)
    global list_dgh
    global list_tree
    if not (list_dgh and list_tree):
        for dgh, tree in DGHs.items():
            list_dgh.append(dgh)
            list_tree.append(tree)
    cost1 = LM_Cost_of_a_table(raw_dataset)
    cost2 = LM_Cost_of_a_table(anonymized_dataset)
    total_LM_cost = abs(cost1 - cost2)

    # print(total_LM_cost)
    return total_LM_cost


def LM_Cost_of_a_node(tree: Tree, node: Node):
    subtree = tree.subtree(node.identifier)
    descendentant_leaves = subtree.leaves(nid=node.identifier)
    number_of_descendants = len(descendentant_leaves)
    total_leaves = tree.leaves(nid='Any')
    number_of_total_nodes = len(total_leaves)
    LM_Cost = (number_of_descendants - 1) / (number_of_total_nodes - 1)
    return LM_Cost


def LM_Cost_of_a_record(record):
    global list_tree
    global list_dgh
    number_of_dghs = len(list_dgh)
    weight = 1.0 / number_of_dghs
    LM_cost_record = 0
    for node, value in record.items():
        if node in list_dgh:
            index = list_dgh.index(node)
            tree = list_tree[index]
            node2 = tree.get_node(value)
            lm_val = LM_Cost_of_a_node(tree, node2)
            LM_cost_record += weight * lm_val

    return LM_cost_record


def LM_Cost_of_a_table(dataset):
    LM_Cost = 0
    for record in dataset:
        LM_Cost += LM_Cost_of_a_record(record)
    return LM_Cost


def randomly_assign_dataset(raw_dataset, k: int, DGHs):
    number_of_clusters = int(len(raw_dataset) / k)
    remainder = int(len(raw_dataset) % k)
    max_number_of_records = int(len(raw_dataset) - remainder)
    record_counter = 0
    dict_of_clustered_records = defaultdict(list)
    for record in raw_dataset:
        rand_val = random.randint(0, number_of_clusters - 1)
        # if not dict_of_clustered_records.get(random)
        if dict_of_clustered_records.get(rand_val):
            while len(dict_of_clustered_records.get(rand_val)) == k:
                if record_counter == max_number_of_records:
                    break
                rand_val = random.randint(0, number_of_clusters - 1)
                if not dict_of_clustered_records.get(rand_val):
                    dict_of_clustered_records[rand_val] = []

            dict_of_clustered_records.get(rand_val).append(record)
        else:
            dict_of_clustered_records[rand_val].append(record)
        record_counter += 1

    lss = {}
    i = 0
    for dgh in DGHs.keys():
        print(dgh)
        lss[i] = dgh
        i += 1
    return dict_of_clustered_records


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
    global DGHs
    if not DGHs:
        DGHs = read_DGHs(DGH_folder)

    anonymized_dataset = []

    # Given a dataset, randomly divide the records in D, into clusters of size k
    # I will use dicts to hold list of records.
    dict_of_clustered_records = randomly_assign_dataset(raw_dataset, k, DGHs)
    number_of_records = int(len(raw_dataset))
    remainder = number_of_records % k
    # Data is clustered into chunks of k records.
    # k-anonymity
    iteration = 1
    k_copy = k
    for equivalence_class in dict_of_clustered_records.values():

        is_last_ec = len(equivalence_class) != k_copy
        if is_last_ec:
            k = k + remainder
        else:
            k = k_copy
        equivalence_class = k_anonymity(equivalence_class, k, DGHs)
        for item in equivalence_class:
            anonymized_dataset.append(item)
        iteration += 1

    write_dataset(anonymized_dataset, output_file)


def k_anon(equivalence_class, DGHs):
    counter = Counter()

    for dgh in DGHs.keys():
        old_dgh = ''
        for item in equivalence_class:
            if old_dgh != '' and item[dgh] != old_dgh:
                return False
    least_common = counter.most_common()[-1]
    return least_common[1]


def is_k_anon(equivalence_class, k, DGHs):
    for dgh in DGHs.keys():
        old_dgh = equivalence_class[0][dgh]
        for item in equivalence_class:
            if item[dgh] != old_dgh:
                return False
    return True


def k_anonymity(equivalence_class, k, DGHs):
    equivalence_class_list = []
    counter = Counter()
    for item in equivalence_class:
        for dgh in DGHs.keys():
            counter[(dgh, item[dgh])] += 1
    for dgh, tree in DGHs.items():
        for item in equivalence_class:
            node = tree.get_node(item[dgh])
            equivalence_class_list.append(node)
    iteration = 1
    for dgh, tree in DGHs.items():
        eq_list = equivalence_class_list[k * (iteration - 1):k * iteration]
        node = get_the_lowest_common_ancestor(tree, eq_list)
        # print(node)
        for item in equivalence_class:
            item[dgh] = node.tag
        iteration += 1

    return equivalence_class


def calculate_dist_of_two_EC(EC1, EC2, DGH_folder: str):
    # Dist
    # temp_EC1_file = 'temp_EC1_file.csv'
    # temp_EC2_file = 'temp_EC2_file.csv'
    # write_dataset(EC1, temp_EC1_file)
    # write_dataset(EC2, temp_EC2_file)
    # dist = cost_MD(temp_EC1_file, temp_EC2_file, DGH_folder)
    dist = MD_cost_of_a_table(EC1, EC2)
    return dist


def find_min_dist(raw_dataset, records_marked_list: list, k_records_list: list, DGH_folder: str, k: int):
    EC_dict = {}
    index = 0
    index_list = []
    copy_records_marked_list = records_marked_list.copy()
    for item in records_marked_list:
        if item == 0:
            index_list.append(index)
        index += 1

    index_holder = 0
    for index in index_list:
        rec = raw_dataset[index]
        # records_marked_list[index] = 1
        EC_dict[index] = k_records_list.copy()
        EC_dict[index].append(rec)
        index_holder += 1
    list_of_dists = []
    temp = iter(EC_dict)
    next_index = next(temp)
    for index, equivalence_class1 in EC_dict.items():
        # if next_index == len(copy_records_marked_list) - 1:
        if next_index == list(EC_dict)[-1]:
            flag = True
            break
        next_index = next(temp)

        equivalence_class2 = EC_dict[next_index]
        dist = calculate_dist_of_two_EC(equivalence_class1, equivalence_class2, DGH_folder)
        list_of_dists.append((index, dist))
    # if flag:
    #     min_val = (index, 0)
    # else:
    if list_of_dists:
        min_val = min(list_of_dists, key=lambda x: x[1])
    else:
        min_val = (index, 0)
    # index_of_min_val = list_of_dists.index((min_val[0],min_val[1]))
    k_records_list = []
    for item in EC_dict[min_val[0]]:
        k_records_list.append(item)
    copy_records_marked_list[min_val[0]] = 1
    records_marked_list = copy_records_marked_list

    return k_records_list, records_marked_list


def cluster_and_assing_dataset(raw_dataset, k: int, DGH_folder: str, DGHs, anonymized_dataset):
    remainder = int(len(raw_dataset) % k)
    record_counter = 0
    max_number_of_records = int(len(raw_dataset) - remainder)
    record_counter = 0

    records_marked_list = [0 for record in raw_dataset]
    iteration = 0
    while records_marked_list.count(0) >= k:
        rec_index = records_marked_list.index(0)
        rec = raw_dataset[rec_index]
        records_marked_list[rec_index] = 1
        k_records_list = [rec]
        record_counter += 1
        for i in range(k - 1):
            # if iteration == max_number_of_records:
            #     continue
            k_records_list, records_marked_list = find_min_dist(raw_dataset, records_marked_list, k_records_list,
                                                                DGH_folder, k)
        k_records_list = k_anonymity(k_records_list, k, DGHs)
        for item in k_records_list:
            anonymized_dataset.append(item)
        iteration += 1

    while records_marked_list.count(0) > 0:
        rec_index = records_marked_list.index(0)
        rec = raw_dataset[rec_index]
        records_marked_list[rec_index] = 1
        k_records_list.append(rec)
        record_counter += 1
    anonymized_dataset = anonymized_dataset[:-k]
    k_records_list = k_anonymity(k_records_list, k + remainder, DGHs)
    for item in k_records_list:
        anonymized_dataset.append(item)

    return anonymized_dataset


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
    global DGHs
    if not DGHs:
        DGHs = read_DGHs(DGH_folder)

    anonymized_dataset = []

    # dist = calculate_dist_of_two_EC("", "", DGH_folder)
    # dict_of_clustered_records = cluster_and_assing_dataset(raw_dataset, k, DGH_folder)
    # for equivalence_class in dict_of_clustered_records.values():
    #     equivalence_class = k_anonymity(equivalence_class, k, DGHs)
    #     for item in equivalence_class:
    #         anonymized_dataset.append(item)
    anonymized_dataset = cluster_and_assing_dataset(raw_dataset, k, DGH_folder, DGHs, anonymized_dataset)

    write_dataset(anonymized_dataset, output_file)
    # os.remove('temp_EC1_file.csv')
    # os.remove('temp_EC2_file.csv')


def get_legal_children(root: Node, k: int):
    global list_dgh
    global list_tree

    root_node = root.tag
    dgh_dict = {}
    for dgh, value in root_node.items():
        index = list_dgh.index(dgh)
        dgh_tree: Tree = list_tree[index]
        current: Node = dgh_tree.get_node(root_node[dgh])
        if not current.is_leaf(dgh_tree.identifier):
            dgh_dict[dgh] = current
    children_nodes_dict = {}
    for dgh, value in dgh_dict.items():
        index = list_dgh.index(dgh)
        dgh_tree: Tree = list_tree[index]
        current: Node = dgh_tree.get_node(value.identifier)
        current_node_successors = current.successors(tree_id=dgh_tree.identifier)
        child_list = []
        dataset = root.data[1]
        for child in current_node_successors:
            child_tag = root.tag.copy()
            child_tag[dgh] = child
            child_list.append((dgh, child))
        if child_list:
            satisfies, lst = satisfies_k_anonymity(dataset, child_list, k)

            if satisfies:
                children_nodes_dict[dgh] = lst

    return children_nodes_dict


def satisfies_k_anonymity(dataset, child_list, k):
    dict = {}
    iteration = 0
    for attribute, val in child_list:
        lst = []
        for x in dataset:
            index1 = list_dgh.index(attribute)
            dgh_tree1: Tree = list_tree[index1]
            subtree = dgh_tree1.subtree(val)
            # if x[attribute] == val:
            if not x[attribute] in subtree:
                continue
            else:
                lst.append(x)

        dict[iteration] = (val, lst)
        iteration += 1
    child_tag_list = []
    for item in dict.values():
        if len(item[1]) < k:
            if not item[1]:
                continue
            return False, []
        child_tag_list.append(item[0])
    return True, child_tag_list


def is_branch_of_the_current_node(child_tag, lst):
    for item in lst:
        for dgh, value in item.items():
            identifier = child_tag[dgh]
            index = list_dgh.index(dgh)
            tree = list_tree[index]
            subtree = tree.subtree(identifier)
            if value in subtree:
                return True
    return False


def specialize(root: Node, k: int, tree: Tree, raw_dataset):
    children_nodes_dict = get_legal_children(root, k)
    LM_cost_list = []
    child_list = []
    for dgh, specialize_list in children_nodes_dict.items():
        # print(dgh)
        LMd = LM_Cost_of_a_record(root.tag)
        LMdns = 0.0
        for i in specialize_list:
            tag = root.tag.copy()
            tag[dgh] = i
            LMdns += LM_Cost_of_a_record(tag)
        LM_cost = abs(LMd - LMdns)
        LM_cost_list.append((specialize_list, LM_cost, dgh))
    if LM_cost_list:
        max_val = max(LM_cost_list, key=lambda x: x[1])

        children = max_val[0]
        dgh_temp = max_val[2]
        # child_list = specialize_list
        for item in children:
            identifier = root.tag.copy()
            identifier[dgh_temp] = item
            lst = get_data_length(raw_dataset, dgh_temp, item)
            data_length = len(lst)

            child_node = tree.create_node(tag=identifier, parent=root, data=(data_length, lst))
            specialize(child_node, k, tree, lst)


def get_data_length(dataset, dgh, identifier):
    dict = {}
    iteration = 0
    attribute = dgh
    val = identifier
    lst = []
    for x in dataset:
        index1 = list_dgh.index(attribute)
        dgh_tree1: Tree = list_tree[index1]
        subtree = dgh_tree1.subtree(val)
        # if x[attribute] == val:
        if not x[attribute] in subtree:
            continue
        else:
            lst.append(x)
    return lst


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
    global DGHs
    if not DGHs:
        DGHs = read_DGHs(DGH_folder)
    global list_dgh
    global list_tree
    if not (list_dgh and list_tree):
        for dgh, tree in DGHs.items():
            list_dgh.append(dgh)
            list_tree.append(tree)
    anonymized_dataset = []
    tree = Tree()
    identifier = []
    data_length = len(raw_dataset)
    tag_dict = {}
    for dgh in DGHs:
        tag_dict[dgh] = 'Any'

    tree.create_node(tag=tag_dict, identifier='root', data=(data_length, raw_dataset))
    root = tree.get_node('root')
    specialize(root, k, tree, raw_dataset=raw_dataset)
    list = tree.leaves()
    for item in list:
        data = item.data[1]
        data_length = item.data[0]
        tag = item.tag
        for element in data:
            for dgh in element:
                if dgh in list_dgh:
                    element[dgh] = tag[dgh]
            anonymized_dataset.append(element)

    # tree.show(key=False)

    write_dataset(anonymized_dataset, output_file)


# print(read_DGHs("DGHs"))
cost_md_temp = cost_MD("adult_small.csv", "adult-clustering-anonymized.csv", "DGHs")
print(cost_md_temp)
# cost_LM("adult_small.csv","adult-random-anonymized.csv", "DGHs" )
# random_anonymizer('adult_small.csv', "DGHs", 8, 'adult-random-anonymized.csv')
# TODO: needs optimization.
# Takes 36 seconds with k = 10, dataset length = 100
# clustering_anonymizer('adult_small.csv', "DGHs", 10, 'adult-clustering-anonymized.csv')
# topdown_anonymizer('adult_small.csv', "DGHs", 10, 'adult-topdown-anonymized.csv')

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
print(f"Results of {k}-anonimity:\n\tCost_MD: {cost_md}\n\tCost_LM: {cost_lm}\n")

# Sample usage:
# python3 code.py clustering DGHs/ adult-hw1.csv result.csv 300
