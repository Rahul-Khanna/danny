"""
    It is often useful to just get information for a single user or a few. Instead of using the full pipeline
    mode of danny, you can just query the index that danny uses in order to obtain the closest user for a
    given user. In order to obtain the index, (the three needed datastructures), you can either use
    supporting_functions or the danny wrapper script.

    Main Functions:
        get_user_neighbors_exact
        get_user_neighbors_approx
        get_user_neighbors_above_thresh
"""
import logging
from operator import itemgetter
import time
from dictionary_based_nn import _strict_prune_space, _approx_prune_space
from dictionary_based_nn import _find_similarities, _format_similarities
from dictionary_based_nn import DEFAULT_USER_CAP

def get_user_neighbors_exact(user_id, user_entity_dict, entity_user_dict, user_entity_matrix,
                             n_neighbors=20, sparse=True):
    """
        Given the three needed data structures that power danny you can quickly get information about any one
        user's information. The three structures act as an index to get information about a user. This
        function uses danny's smart, but comprehensive mode to find nearest neighbors. You can read more
        about this mode in dictionary_based_nn.

        Params:
            user_id               (int) : id of user whose nearest neighbors are wanted
            user_entity_dict     (dict) : dictionary where: key – user_id | value - list of entity ids user
                                          has visited
            entity_user_dict     (dict) : dictionary where: key - entity_id | value - list of user ids that
                                          have visited the entity
            user_entity_matrix (matrix) : matrix where each row contains a user's entity visitation pattern
                                          encoded as a count vector. Each column contains each entity's user
                                          vistiation record encoded as a count vector. Can be sparse.
            n_neighbors           (int) : number of neighbors requested
            sparse               (bool) : is the matrix a sparse one or not

        Returns:
            arr : each element is a tuple (user_id, dot_product) representing the n closest neighbors in
                  order from closest to furthest
    """
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

    results_dict = _format_similarities(users_to_compare_to, results)

    nearest_neighbors = sorted(results_dict.items(), key=itemgetter(1), reverse=True)[:n_neighbors]
    logging.info("Took %s seconds to get number of requested neighbors", time.time() - start_time)

    return nearest_neighbors

def get_user_neighbors_approx(user_id, user_entity_dict, entity_user_dict, user_entity_matrix,
                              n_neighbors=20, sparse=True):
    """
        Given the three needed data structures that power danny you can quickly get information about any one
        user's given information. The three structures act as an index to get information about a user. This
        function uses danny's approximate mode to find nearest neighbors. You can read more
        about this mode in dictionary_based_nn.
            * In practice I have seen this work faster than its exact counterpart

        Params:
            user_id               (int) : id of user whose nearest neighbors are wanted
            user_entity_dict     (dict) : dictionary where: key – user_id | value - list of entity ids user
                                          has visited
            entity_user_dict     (dict) : dictionary where: key - entity_id | value - list of user ids that
                                          have visited the entity
            user_entity_matrix (matrix) : matrix where each row contains a user's entity visitation pattern
                                          encoded as a count vector. Each column contains each entity's user
                                          vistiation record encoded as a count vector. Can be sparse.
            n_neighbors           (int) : number of neighbors requested
            sparse               (bool) : is the matrix a sparse one or not

        Returns:
            arr : each element is a tuple (user_id, dot_product) representing the n closest neighbors in
                  order
    """
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
    logging.info("Took %s seconds to get number of requested neighbors", time.time() - start_time)

    return nearest_neighbors

def get_user_neighbors_above_thresh(user_id, user_entity_dict, entity_user_dict, user_entity_matrix,
                                    thresh=0.9, sparse=True, sort=False):
    """
        Given the three needed data structures that power danny you can quickly get information about any one
        user's given information. The three structures act as an index to get information about a user. This
        function returns all users who are within a certain distance of a user.

        Params:
            user_id               (int) : id of user whose nearest neighbors are wanted
            user_entity_dict     (dict) : dictionary where: key – user_id | value - list of entity ids user
                                          has visited
            entity_user_dict     (dict) : dictionary where: key - entity_id | value - list of user ids that
                                          have visited the entity
            user_entity_matrix (matrix) : matrix where each row contains a user's entity visitation pattern
                                          encoded as a count vector. Each column contains each entity's user
                                          vistiation record encoded as a count vector. Can be sparse.
            thresh              (float) : minimum dot product required for a user to be considered close
            sparse               (bool) : is the matrix a sparse one or not
            sort                 (bool) : return the users sorted by the closest to furthest

        Returns:
            arr : each element is a tuple (user_id, dot_product) representing all neighbors within a certain
                  distance from the passed in user
    """
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
    
    logging.info("Took %s seconds to get number of requested neighbors", time.time() - start_time)

    return nearest_neighbors
