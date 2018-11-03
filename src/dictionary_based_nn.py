import logging
from multiprocessing import Pool, cpu_count
from operator import itemgetter
import time
from numpy import dot
from supporting_functions import read_pickle_file
from supporting_functions import write_pickle_file

DEFAULT_DIR = "output_data/"
MAX_PROCESSES = cpu_count()
DEFAULT_USER_CAP = 500
SUM_SIGNIFICANCE = 10
USER_ENTITY_DICT = None
ENTITY_USER_DICT = None
USER_ENTITY_MATRIX = None

def _update_score(perc):
    if perc > 0.5:
        return 2
    if perc > 0.25:
        return 1
    return 0.5

def _get_top_n_users_batch(user_id_user_cap):
    user_id = user_id_user_cap[0]
    user_cap = user_id_user_cap[1]

    users_to_look_at = {}
    user_sum = sum(USER_ENTITY_DICT[user_id].values())
    is_sum_sig = user_sum > SUM_SIGNIFICANCE
    for entity in USER_ENTITY_DICT[user_id]:
        users_associated_with_entity = ENTITY_USER_DICT[entity]
        if is_sum_sig:
            user_count = USER_ENTITY_DICT[user_id][entity]
            perc = user_count / user_sum
        for key in users_associated_with_entity:
            if key in users_to_look_at:
                if is_sum_sig:
                    users_to_look_at[key] += _update_score(perc)
                else:
                    users_to_look_at[key] += 1
            else:
                if is_sum_sig:
                    users_to_look_at[key] = _update_score(perc)
                else:
                    users_to_look_at[key] = 1

    top_n = sorted(users_to_look_at.items(), key=itemgetter(1), reverse=True)[:user_cap]

    top_n_keys = [tup[0] for tup in top_n]

    return top_n_keys

def _get_relevant_users_batch(user_id):
    users_to_look_at = {}
    for entity in USER_ENTITY_DICT[user_id]:
        users_associated_with_entity = ENTITY_USER_DICT[entity]
        for key in users_associated_with_entity:
            users_to_look_at[key] = 1

    return list(users_to_look_at.keys())

def _get_dense_similarities_batch(user_tuple):
    user_id = user_tuple[0]
    users_to_compare_to = user_tuple[1]
    row = USER_ENTITY_MATRIX[user_id]

    results = dot(USER_ENTITY_MATRIX[users_to_compare_to, :], row)

    results_dict = {}
    for i, dot_product in enumerate(results):
        results_dict[users_to_compare_to[i]] = float("{0:.4f}".format(dot_product))

    return results_dict

def _get_sparse_similarities_batch(user_tuple):
    user_id = user_tuple[0]
    users_to_compare_to = user_tuple[1]
    row = USER_ENTITY_MATRIX[user_id].toarray()[0]

    results = USER_ENTITY_MATRIX[users_to_compare_to, :].dot(row)

    results_dict = {}
    for i, dot_product in enumerate(results):
        results_dict[users_to_compare_to[i]] = float("{0:.4f}".format(dot_product))

    return results_dict


def get_nearest_neighbors_batch(input_type="default", file_names=None, sparse=True, user_cap=DEFAULT_USER_CAP,
                                n_processes=None, save=True, output_dir=DEFAULT_DIR):
    # pylint: disable=unused-variable, global-statement, too-many-arguments, too-many-locals
    start_time = time.time()
    input_types = ["default", "files"]
    if input_type not in input_types:
        raise ValueError("input_type must be \"default\" or \"files\"")

    if input_type == "files" and (not isinstance(file_names, list) or len(file_names) != 3):
        raise ValueError("an array of length 3 must be passed in for \"file_names\" indicating \
            the path+filename of the two dictionary files and the matrix file. The order of the \
            files should be: 1. user_entity_dict, 2. entity_user_dict, 3. user_entity_matrix")

    if n_processes is not None and (not isinstance(n_processes, int) or n_processes > MAX_PROCESSES):
        raise ValueError("n_processes must be an int smaller than {}, as your computer only has {} \
            cores".format(MAX_PROCESSES, MAX_PROCESSES))

    user_entity_dict_file_name = file_names[0] if input_type == "files" else \
                                 output_dir + "user_entity_dict.pickle"
    entity_user_dict_file_name = file_names[1] if input_type == "files" else \
                                 output_dir + "entity_user_dict.pickle"
    user_entity_matrix_file_name = file_names[2] if input_type == "files" else \
                                   output_dir + "user_entity_matrix.pickle"

    global USER_ENTITY_DICT, ENTITY_USER_DICT, USER_ENTITY_MATRIX

    USER_ENTITY_DICT = read_pickle_file(user_entity_dict_file_name)
    ENTITY_USER_DICT = read_pickle_file(entity_user_dict_file_name)
    USER_ENTITY_MATRIX = read_pickle_file(user_entity_matrix_file_name)
    logging.info("read in needed pickle files in %s seconds", time.time() - start_time)
    start_time = time.time()

    if user_cap > 0:
        user_indicies = []
        for key in USER_ENTITY_DICT:
            user_indicies.append((key, user_cap))
    else:
        user_indicies = list(USER_ENTITY_DICT.keys())

    logging.info("prepped users to be analyzed in %s seconds", time.time() - start_time)
    start_time = time.time()

    n_processes = MAX_PROCESSES - 2 if n_processes is None else n_processes

    pool = Pool(processes=n_processes)

    users_to_extract = pool.map(_get_top_n_users_batch, user_indicies) if user_cap \
                       else pool.map(_get_relevant_users_batch, user_indicies)

    logging.info("Pruning took %s seconds", time.time() - start_time)
    start_time = time.time()

    user_tuples = []
    for i, users in enumerate(users_to_extract):
        user_tuples.append((i, users))

    logging.info("prepped users for dot products in %s seconds", time.time() - start_time)
    start_time = time.time()

    dictionary_results = pool.map(_get_sparse_similarities_batch, user_tuples) if sparse \
                         else pool.map(_get_dense_similarities_batch, user_tuples)

    logging.info("Matrix Multiplications took %s seconds", time.time() - start_time)

    pool.close()
    pool.join()

    similarity_scores = {}

    for i, dict_result in enumerate(dictionary_results):
        similarity_scores[i] = dict_result

    if save:
        similarity_scores_file_name = output_dir + "similarity_scores.pickle"
        write_pickle_file(similarity_scores, similarity_scores_file_name)

        del similarity_scores
        return True
    
    return similarity_scores


def get_user_neighbors_exact(user_id, user_entity_dict, entity_user_dict, user_entity_matrix,
                             n_neighbors=20, sparse=True):
    # pylint: disable=too-many-arguments, too-many-locals
    if user_id not in user_entity_dict:
        raise ValueError("The user_id passed in is not found in the user_entity_dict")

    start_time = time.time()
    relevant_users = {}
    for entity in user_entity_dict[user_id]:
        for user in entity_user_dict[entity]:
            relevant_users[user] = 1

    users_to_compare_to = list(relevant_users.keys())
    row = user_entity_matrix[user_id]

    logging.info("Took %s seconds to prune search space", time.time() - start_time)
    start_time = time.time()
    
    if sparse:
        results = user_entity_matrix[users_to_compare_to, :].dot(row)
    else:
        results = dot(user_entity_matrix[users_to_compare_to, :], row)

    logging.info("Took %s seconds to preform needed dot products", time.time() - start_time)
    start_time = time.time()

    results_dict = {}
    for i, dot_product in enumerate(results):
        results_dict[users_to_compare_to[i]] = float("{0:.4f}".format(dot_product))

    nearest_neighbors = sorted(results_dict.items(), key=itemgetter(1), reverse=True)[:n_neighbors]
    logging.info("Tool %s seconds to get number of requested neighbords", time.time() - start_time)

    return nearest_neighbors

def get_user_neighbors_approx(user_id, user_entity_dict, entity_user_dict, user_entity_matrix,
                              n_neighbors=20, sparse=True):
    # pylint: disable=too-many-arguments, too-many-locals, too-many-branches
    if user_id not in user_entity_dict:
        raise ValueError("The user_id passed in is not found in the user_entity_dict")

    start_time = time.time()
    relevant_users = {}
    user_sum = sum(user_entity_dict[user_id].values())
    is_sum_sig = user_sum > SUM_SIGNIFICANCE
    for entity in user_entity_dict[user_id]:
        if is_sum_sig:
            user_count = user_entity_dict[user_id][entity]
            perc = user_count / user_sum
        for user in entity_user_dict[entity]:
            if user in relevant_users:
                if is_sum_sig:
                    relevant_users[user] += _update_score(perc)
                else:
                    relevant_users[user] += 1
            else:
                if is_sum_sig:
                    relevant_users[user] = _update_score(perc)
                else:
                    relevant_users[user] = 1

    if len(relevant_users) > n_neighbors * 2:
        most_likely_users = sorted(relevant_users.items(), key=itemgetter(1), reverse=True)[:n_neighbors*2]
        users_to_compare_to = [tup[0] for tup in most_likely_users]
    else:
        users_to_compare_to = list(relevant_users.keys())

    logging.info("Took %s seconds to prune search space", time.time() - start_time)
    start_time = time.time()

    row = user_entity_matrix[user_id]

    if sparse:
        results = user_entity_matrix[users_to_compare_to, :].dot(row)
    else:
        results = dot(user_entity_matrix[users_to_compare_to, :], row)

    logging.info("Took %s seconds to preform needed dot products", time.time() - start_time)
    start_time = time.time()

    results_dict = {}
    for i, dot_product in enumerate(results):
        results_dict[users_to_compare_to[i]] = float("{0:.4f}".format(dot_product))

    nearest_neighbors = sorted(results_dict.items(), key=itemgetter(1), reverse=True)[:n_neighbors]
    logging.info("Tool %s seconds to get number of requested neighbords", time.time() - start_time)

    return nearest_neighbors
