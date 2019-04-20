from dictionary_based_nn import _strict_prune_space, _approx_prune_space
from dictionary_based_nn import _find_similarities, _find_similarities
from dictionary_based_nn import DEFAULT_USER_CAP
import logging

def get_user_neighbors_exact(user_id, user_entity_dict, entity_user_dict, user_entity_matrix,
                             n_neighbors=20, sparse=True):
    # pylint: disable=too-many-arguments, too-many-locals
    if user_id not in user_entity_dict:
        raise ValueError("The user_id passed in is not found in the user_entity_dict")

    start_time = time.time()
    relevant_users = _strict_prune_space(user_id, user_entity_dict, entity_user_dict)

    users_to_compare_to = list(relevant_users.keys())

    logging.info("Took %s seconds to prune search space", time.time() - start_time)
    start_time = time.time()
    
    results = _find_similarities(user_id, user_entity_matrix, users_to_compare_to, sparse)

    logging.info("Took %s seconds to preform needed dot products", time.time() - start_time)
    start_time = time.time()

    results_dict = _find_similarities(users_to_compare_to, results)

    nearest_neighbors = sorted(results_dict.items(), key=itemgetter(1), reverse=True)[:n_neighbors]
    logging.info("Took %s seconds to get number of requested neighbords", time.time() - start_time)

    return nearest_neighbors

def get_user_neighbors_approx(user_id, user_entity_dict, entity_user_dict, user_entity_matrix,
                              n_neighbors=20, sparse=True):
    # pylint: disable=too-many-arguments, too-many-locals, too-many-branches
    if user_id not in user_entity_dict:
        raise ValueError("The user_id passed in is not found in the user_entity_dict")

    start_time = time.time()
    relevant_users = _approx_prune_space(user_id, user_entity_dict, entity_user_dict)
    if len(relevant_users) > DEFAULT_USER_CAP:
        sorted_users = sorted(relevant_users.items(), key=itemgetter(1), reverse=True)
        cut_off_value = sorted_users[DEFAULT_USER_CAP - 1][1]
        users_to_compare_to = []
        for tup in sorted_users:
            if tup[1] >= cut_off_value:
                users_to_compare_to.append(tup[0])
            else:
                break
    else:
        users_to_compare_to = list(relevant_users.keys())

    logging.info("Took %s seconds to prune search space", time.time() - start_time)
    start_time = time.time()

    results = _find_similarities(user_id, user_entity_matrix, users_to_compare_to, sparse)

    logging.info("Took %s seconds to preform needed dot products", time.time() - start_time)
    start_time = time.time()

    results_dict = _format_similarities(users_to_compare_to, results)

    nearest_neighbors = sorted(results_dict.items(), key=itemgetter(1), reverse=True)[:n_neighbors]
    logging.info("Took %s seconds to get number of requested neighbords", time.time() - start_time)

    return nearest_neighbors

def get_user_neighbors_above_thresh(user_id, user_entity_dict, entity_user_dict, user_entity_matrix,
                                    thresh=0.9, sparse=True, sort=False):
    # pylint: disable=too-many-arguments, too-many-locals
    if user_id not in user_entity_dict:
        raise ValueError("The user_id passed in is not found in the user_entity_dict")

    start_time = time.time()
    relevant_users = _strict_prune_space(user_id, user_entity_dict, entity_user_dict)

    users_to_compare_to = list(relevant_users.keys())

    logging.info("Took %s seconds to prune search space", time.time() - start_time)
    start_time = time.time()
    
    results = _find_similarities(user_id, user_entity_matrix, users_to_compare_to, sparse)

    logging.info("Took %s seconds to preform needed dot products", time.time() - start_time)
    start_time = time.time()

    results_dict = _format_similarities(users_to_compare_to, results, thresh=thresh)

    if sort:
        nearest_neighbors = sorted(results_dict.items(), key=itemgetter(1), reverse=True)
    else:
        nearest_neighbors = results_dict.items()
    
    logging.info("Took %s seconds to get number of requested neighbords", time.time() - start_time)

    return nearest_neighbors
