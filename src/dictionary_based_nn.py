import pickle
from numpy import dot
from operator import itemgetter
from multiprocessing import Pool, cpu_count
import time
from supporting_functions import read_pickle_file
from supporting_functions import write_pickle_file

DEFAULT_DIR = "output_data/"
MAX_PROCESSES = cpu_count()
global USER_ENTITY_DICT, ENTITY_USER_DICT, USER_ENTITY_MATRIX

def _get_top_n_users(user_id_user_cap):
	user_id = user_id_user_cap[0]
	user_cap = user_id_user_cap[1]

    users_to_look_at = {}
    user_sum = sum(USER_ENTITY_DICT[user_id].values())
    is_sum_sig = user_sum > 10
    for entity in USER_ENTITY_DICT[user_id]:
        users_associated_with_entity = ENTITY_USER_DICT[entity]
        if is_sum_sig:
            user_count = USER_ENTITY_DICT[user_id][entity]
            perc = user_count / user_sum
        for key in users_associated_with_domain:
            if key in users_to_look_at:
                if is_sum_sig:
                    if perc > 0.5:
                        users_to_look_at[key] += 2
                    if perc > 0.25:
                        users_to_look_at[key] += 1
                    else:
                        users_to_look_at[key] += 0.5
                else:
                    users_to_look_at[key] += 1
            else:
                if is_sum_sig:
                    if perc > 0.5:
                        users_to_look_at[key] = 2
                    if perc > 0.25:
                        users_to_look_at[key] = 1
                    else:
                        users_to_look_at[key] = 0.5
                else:
                    users_to_look_at[key] = 1

    top_N = sorted(users_to_look_at.items(), key=itemgetter(1), reverse=True)[:user_cap]

    top_N_keys = [tup[0] for tup in top_N]

    return top_N_keys

def _get_relevant_users(user_id):
    users_to_look_at = {}
    for entity in USER_ENTITY_DICT[user_id]:
        users_associated_with_entity = ENTITY_USER_DICT[entity]
        for key in users_associated_with_domain:
            users_to_look_at[key] = 1

    return list(users_to_look_at.keys())

def _get_dense_similarities(user_tuple):
	user_id = user_tuple[0]
    users_to_compare_to = user_tuple[1]
    row = USER_ENTITY_MATRIX[user_id]

    results = dot(main_matrix[users_to_compare_to, :], row)

    results_dict = {}
    for i, dot_product in enumerate(results):
        results_dict[users_to_compare_to[i]] = float("{0:.4f}".format(dot_product))

    return results_dict

def _get_sparse_similarities(user_tuple):
    user_id = user_tuple[0]
    users_to_compare_to = user_tuple[1]
    row = USER_ENTITY_MATRIX[user_id].toarray()[0]

    results = USER_ENTITY_MATRIX[users_to_compare_to, :].dot(row)

    results_dict = {}
    for i, dot_product in enumerate(results):
        results_dict[users_to_compare_to[i]] = float("{0:.4f}".format(dot_product))

    return results_dict

def get_nearest_neighbors(input_type="default", file_names=None, sparse=True, save=True
	                      output_dir=DEFAULT_DIR, user_cap=500, n_processes=None):

	input_types = ["default", "files"]
	if input_types not in input_types:
		raise ValueError("input_type must be \"default\" or \"files\"")

	if input_type == "files" and (not isinstance(file_names, list) or len(file_names) != 3):
		raise ValueError("an array of length 3 must be passed in for \"file_names\" indicating \
			the path+filename of the two dictionary files and the matrix file. The order of the \
			files should be: 1. user_entity_dict, 2. entity_user_dict, 3. user_entity_matrix")

	if n_processes not None and (not isinstance(n_processes, int) or n_processes > MAX_PROCESSES):
		raise ValueError("n_processes must be an int smaller than {}, as your computer only has {} \
			cores".format(MAX_PROCESSES, MAX_PROCESSES))

	user_entity_dict_file_name = file_names[0] if input_type == "files" else /
	                             output_dir + "user_entity_dict.pickle"
	entity_user_dict_file_name = file_names[1] if input_type == "files" else /
	                             output_dir + "entity_user_dict.pickle"
	user_entity_matrix_file_name = file_names[2] if input_type == "files" else /
	                               output_dir + "user_entity_matrix.pickle"

	USER_ENTITY_DICT = read_pickle_file(user_dict_file_name)
	ENTITY_USER_DICT = read_pickle_file(entity_user_dict_file_name)
	USER_ENTITY_MATRIX = read_pickle_file(user_entity_matrix_file_name)

	if user_cap > 0:
		user_indicies = []
		for key in user_entity_dict:
			user_indicies[(key, user_cap)]
	else:
		user_indicies = list(user_entity_matrix.keys())

	n_processes = MAX_PROCESSES - 2 if n_processes is None else n_processes

	pool = Pool(processes=n_processes)

	start_time = time.time()

	users_to_extract = pool.map(_get_top_n_users, user_indicies) if user_cap /
	                   else pool.map(_get_relevant_users, user_indicies)

	print("Pruning took {0:.3f} seconds".format(time.time() - start_time))

	user_tuples = []
	for i, users in enumerate(users_to_extract):
    	user_tuples.append((i, users))

	start_time = time.time()

	dictionary_results = pool.map(_get_sparse_similarities, user_tuples) if sparse /
	                     else pool.map(_get_dense_similarities)

	print("Matrix Multiplications took {0:.3f} seconds".format(time.time() - start_time))

	pool.close()
	pool.join()

	final_results = {}

	for i, dict_result in enumerate(dictionary_results):
    	final_results[i] = dict_result

    if save:
    	similarity_scores_file_name = output_dir + "similarity_scores.pickle"
    	write_pickle_file(final_results, similarity_scores_file_name)

    	del final_results
    else:
    	return similarity_scores
