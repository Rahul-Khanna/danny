"""
    The core functionalities of danny are written here. Given three specific data structures, danny will
    parralelize the task of finding nearest neighbors per each user. Right now danny does this using Python's
    multiprocessing library, but these functions could be easily ported over to a Map Reduce job.

    The three needed data structures are:
        1. user-entity dictionary: key - user_id | value - dictionary
            sub dictionary: key - entity_id | value - either total visits by user_id to entity_id
                                                      or 1 for one hot encoding
        2. entity-user dictionary: key - entity_id | value - dictionary
            sub dictionary: key - user_id | value - either total visits by user_id to entity_id
                                                    or 1 for one hot encoding
        3. user-entity matrix:
            * can be sparse or dense
            * row_index = user_index (x_i)
            * column_index = entity_index (y_i)
            * M[x_i][y_i] = either total visits by user_id to entity_id or 1 for one hot encoding

    All three can be obtained using the supporting_functions module, or through the danny wrapper script.
    Parts of these can also be easily written as a Map Reduce job, and are using Python's multiprocessing
    library.

    There are two seperate strategies danny can use in order to speed up the search for nearest neighbors:
        * a smart, but more comprehensive search through the user space.
        * an approximate mode (default)

    Smart, but more comprehensive mode:
        Danny's core idea is centered around the idea of reducing the search space for all users by using
        more precomputed data in a parralelized manner. By using dictionaries it is easy to quickly reduce
        the search space per user to only those users who share a common entity. If user_i and user_j do not
        share a common entity it is very difficult to consider user_i's entity vistiation pattern and
        user_j's visitation pattern as similar.

        Once the search space per user is reduced to only those users who share a common entity, we can
        search through this space using matrix mulltiplication, but with a much smaller number of
        calculations needed per user. This is done in _find_similarities.

        To use this functionality you must call get_nearest_neighbors_batch with user_cap=-1

        Let G, be a bi-partite graph G(U, V, E), where U = set of all users, V = set of all entities,
        E = set of all edges
            * For reference, general time complexity of NN is O(|U|^2*|V|)

        Average Time complexity:
            let k = average number of users per user who share a common entity

            * Pruning the space: O(|U| * avg_d(u) * avg_d(v)) = O(|U|*(|E|/|U|)*(|E|/|V|)) = O(|E|^2/|V|)
           
            * Matrix mulltiplication: O(|U|*|V|*k)

            So when O(|E|) << O(|U|*|V|): (Note: |E| is maxed out at |U|*|V| in a bipartite graph)

                * O(|E|^2) <<<< O((|U|*|V|)^2)
                * therefore: O(|E|^2/|V|) <<<< O((|U|*|V|)^2 / |V|)
                * but (|U|*|V|)^2 / |V| == |U|^2*|V|
                * therefore: O(|E|^2/|V|) <<<< O(|U|^2*|V|)

                * also if O(k) << O(|U|)
                    * which is highly likley given the degrees of each vertex (user and entity) is low
                * then O(|U|*|V|*k) ~ O(|U|*|V|) << O(|U|^2*|V|)

    Approximate mode:
        Building on top of the reducing the search space idea, instead of just looking for users who share a
        common entity, danny assigns a "closeness score" to each user that shares an entity with a given user
        . danny then selects the top-n users from this list of users with common entities, where n by default
        is 500, but is configurable. These top-n users are the only users who will be compared to a given
        user in the matrix multiplication step. Extra time is spent here as we sort the list of users with a
        common entity, but it is justified as we can set a cap on the number of dot products required per
        user without loosing too much accuracy.

        Let G, be a bi-partite graph G(U, V, E), where U = set of all users, V = set of all entities,
        E = set of all edges
            * For reference, general time complexity of NN is O(|U|^2*|V|)

        Average Time complexity:
            For reference, general time complexity of NN is O(|U|^2*|V|)

            let k = average number of users per user who share a common entity

            * Pruning the space: O(|U| * (avg_d(u) * avg_d(v) + klog(k)))
                               = O(|U|*((|E|/|U|)*(|E|/|V|) + klog(k)))
                               = O(|E|^2/|V| + |U|*klog(k))
           
            * Matrix mulltiplication: O(|U|*|V|)

            So when O(|E|) << O(|U|*|V|): (Note: |E| is maxed out at |U|*|V| in a bipartite graph)

                * if O(k) << O(|U|)
                    * which is highly likley given the degrees of each vertex (user and entity) is low
                * then O(|U|*klog(k)) ~ O(|U|)
                * then O(|E|^2/|V| + |U|*klog(k)) ~ O(|E|^2/|V| + |U|)
                * but O(|E|^2/|V|) <<<< O(|U|^2*|V|) (shown above)
                * and |U| <= |E| <= |E|^2/|E| <= |E|^2/|V|
                * therfore  O(|E|^2/|V| + |U|) <<<< O(|U|^2*|V|)

                * O(|U|*|V|) << O(|U|^2*|V|)

    You can also just prune your search space in a parralelized way, or just get the matrix mulltiplications
    done in a parralelized way if desired.

    Important functions:
        prune_space_batch
        matrix_multiplication_batch
        get_nearest_neighbors_batch
"""
import gc
import logging
from multiprocessing import Pool, cpu_count
from operator import itemgetter
import random
import time
from numpy import dot
from supporting_functions import read_pickle_file
from supporting_functions import write_pickle_file

DEFAULT_DIR = "output_data/"
MAX_PROCESSES = cpu_count()
DEFAULT_USER_CAP = 500
MAX_USER_CAP = 1000
SUM_SIGNIFICANCE = 10

USER_ENTITY_DICT = None
ENTITY_USER_DICT = None
USER_ENTITY_MATRIX = None

def _update_score(perc, number_of_entities_user_1, number_of_entities_user_2):
    """
        Heuristic used to determine how similar user_2's entity visitation pattern is to user_1's entity
        visitation pattern. The main idea behind the function is to look at:
            * how significant the current entity is in user_1's visitation history (the perc param)
            * the difference in the number of entities user_1 and user_2 have visited

        The more significant the current entity is in user_1's history, and the smaller the difference in the
        number of entities visited between the two users, the more likely the two users are going to have
        similar visitation patterns.

        Note: This function is only called once we have established that user_1 and user_2 have both visited
              the entity in question

        Params:
            perc                    (float) : percentage of user_1's total visits that are attributed to
                                              entity_a
            number_of_entities_user_1 (int) : number of entities user_1 has visited
            number_of_entities_user_2 (int) : number of entities user_2 has visited

        Returns:
            float : score to update how likely user_2's visitation pattern is close to user_1's
                    visitation pattern

    """
    return perc / (abs(number_of_entities_user_1 - number_of_entities_user_2) + 1)

def _approx_prune_space(user_id, user_entity_dict, entity_user_dict):
    """
        Function called by the pool workers in order to establish which users are most likely to have close
        entity visitation patterns to the user with the passed in user_id.

        * The crux of what danny does is it throws out all users who do not share an entity with the user in
          question.
        * This function takes that one step further. Of the users who share an entity with the
          user-in-question (user associated with the user_id) the function assigns a score as to how likely
          each user's vistiation pattern is to the user-in-question's visitation pattern
        * It only attempts to create this score per user if the user-in-question has had enough overall
          visits––the benefits of the extra calculations to determine this score are not realized if the
          user-in-question is not a high volume user
       
        Params:
            user_id           (int) : id of the user whose list of potential close users is needed
            user_entity_dict (dict) : dictionary where: key – user_id | value - list of entity ids user
                                      has visited
            entity_user_dict (dict) : dictionary where: key - entity_id | value - list of user ids that
                                      have visited the entity

        Returns:
            dict : dictionary containing users to look at when evaluating which users are closest to the user
                   associated with the user_id
                   key - user_id | value - score of how likely that user's visitation patterns is close to
                                           user-in-question's visitation patterns
    """
    users_to_look_at = {}
    user_sum = sum(user_entity_dict[user_id].values())
    is_sum_sig = user_sum > SUM_SIGNIFICANCE
    user_length = len(user_entity_dict[user_id])
   
    for entity in user_entity_dict[user_id]:
        users_associated_with_entity = entity_user_dict[entity]
        if is_sum_sig:
            user_count = user_entity_dict[user_id][entity]
            perc = user_count / user_sum
        for key in users_associated_with_entity:
            if key in users_to_look_at:
                if is_sum_sig:
                    users_to_look_at[key] += _update_score(perc, user_length, len(user_entity_dict[key]))
                else:
                    users_to_look_at[key] += 1
            else:
                if is_sum_sig:
                    users_to_look_at[key] = _update_score(perc, user_length, len(user_entity_dict[key]))
                else:
                    users_to_look_at[key] = 1

    return users_to_look_at

def _strict_prune_space(user_id, user_entity_dict, entity_user_dict):
    """
        Function called by the pool workers in order to establish which users have any chance of having close
        entity visitation patterns to the user with the passed in user_id.

        The crux of what danny does is it throws out all users who do not share an entity with the user in
        question. It accomplishes this by looking at each entity a user has visited and then grabbing all
        users who have visited these entities. If user_i has not shared an entity with a user_j, then there's
        very little sense in looking at how close user_i and user_j are.
       
        Params:
            user_id           (int) : id of the user whose list of potential close users is needed
            user_entity_dict (dict) : dictionary where: key – user_id | value - list of entity ids user
                                      has visited
            entity_user_dict (dict) : dictionary where: key - entity_id | value - list of user ids that
                                      have visited the entity

        Returns:
            dict : dictionary containing users to look at when evaluating which users are closest to the user
                   associated with the user_id, key - user_id | value - 1
    """
    users_to_look_at = {}
   
    for entity in user_entity_dict[user_id]:
        users_associated_with_entity = entity_user_dict[entity]
        for key in users_associated_with_entity:
            users_to_look_at[key] = 1
   
    return users_to_look_at

def _get_top_n_users_batch(user_id_user_cap):
    """
        Actual function called by pool workers to pick out the top n most likely users similar to a given
        user (user-in-question) for the final vector comparison. This function is called when danny is
        running in approximate mode.

        Function first calls _approx_prune_space, which assings a scores (via a heuristic) to every relevant
        user. This score is a measure of how likely each relevant user's visitation patterns are to the
        user-in-question's visitation pattern.
            * A relevant user is a user whose visitation pattern shares at least one entity with the
              user-in-question's visitation pattern.

        Next the function sorts the relevant users by their scores, and then selects the top n users, where
        n is passed in (user_cap). If there is a tie in scores that causes the list to expand past n users,
        the function randomly samples from the tied users and ensures the list is below the max number of
        associated users per each user-in-question. The max number is 1000 and this is done for storage
        purposes.

        Params:
            user_id_user_cap (tup) : user_id, number of similar users desired (max is 1000)

        Returns:
            tup : user_id, list of n most similar users

    """
    user_id = user_id_user_cap[0]
    user_cap = min(user_id_user_cap[1], MAX_USER_CAP)

    users_to_look_at = _approx_prune_space(user_id, USER_ENTITY_DICT, ENTITY_USER_DICT)
   
    if len(users_to_look_at) > user_cap:
        sorted_users = sorted(users_to_look_at.items(), key=itemgetter(1), reverse=True)
        cut_off_value = sorted_users[user_cap - 1][1]

        top_n_keys = []
        keys_to_randomly_select_from = []
   
        for tup in sorted_users:
            if tup[1] > cut_off_value:
                top_n_keys.append(tup[0])
            elif tup[1] == cut_off_value:
                keys_to_randomly_select_from.append(tup[0])
            else:
                break

        sample_amount = min(MAX_USER_CAP - len(top_n_keys), len(keys_to_randomly_select_from))
        if sample_amount == len(keys_to_randomly_select_from):
            keys_to_add = keys_to_randomly_select_from
        else:
            keys_to_add = random.sample(keys_to_randomly_select_from, sample_amount)
       
        for key in keys_to_add:
            top_n_keys.append(key)
    else:
        top_n_keys = list(users_to_look_at.keys())

    return (user_id, top_n_keys)

def _get_relevant_users_batch(user_id):
    """
        Actual function called by pool workers to pick out all relevant users when considering which
        users would have close visitation patterns. This function is called when danny is
        running in smart, but more comprehensive mode.

        Params:
            user_id (int) : id of user to grab relevant users for

        Returns:
            tup : user_id, list of relevant user_ids

    """
    users_to_look_at = _strict_prune_space(user_id, USER_ENTITY_DICT, ENTITY_USER_DICT)

    return (user_id, [key for key in users_to_look_at])

def _find_similarities(user_id, matrix, users_to_compare_to, sparse):
    """
        Function called by pool workers to computes the dot product between user-in-question (user associated
        with user_id) and the list of user_ids passed in.

        Note: user_ids (which are ints) must equal the row indicies associated with the count vectors for
              those user_ids. i.e. if a user's id is 0, then that user's count vector must be stored in row
              zero of the passed in matrix.
       
        Params:
            user_id             (int) : user whose similar users are wanted
            matrix           (matrix) : matrix where each row contains a user's entity visitation pattern
                                        encoded as a count vector. Each column contains each entity's user
                                        vistiation record encoded as a count vector. Can be sparse.
            users_to_compare_to (arr) : list of user_ids whose entity visitation patterns should be compared
                                        to the user-in-question's entity visitation pattern
            sparse             (bool) : is the matrix sparse or not

        Returns:
            arr : similarities between user-in-question and passed in user_ids, order is preserved between
                  the passed in user_ids and the returned dot products
    """
    if sparse:
        row = matrix[user_id].toarray()[0]
        results = matrix[users_to_compare_to, :].dot(row)
    else:
        row = matrix[user_id]
        results = dot(matrix[users_to_compare_to, :], row)

    return results

def _format_similarities(users_to_compare_to, similarities, thresh=-1.0):
    """
        Both for readability and storage purposes it is useful to cap the number of decimal places for the
        similarities scores (dot products) computed for a user and their closest users. I have chosen 4, this
        decision is arbirtrary though.

        As no additional complexity is added to the process, there is an ability to filter out similarities
        that are below a certain threshold when preparring similairty scores.

        This function is also called by pool workers.

        Params:
            users_to_compare_to (arr) : user_ids asscoiated with the dot scores that are being
                                        formated. The function assumes that user_id in position i is
                                        asscoiated with similarity score in position i
            similarities        (arr) : similarity scores to be formatted
            thresh            (float) : minimum threshold for scores to be formatted

        Returns:
            dict : key - user_id | value - formatted similarity score

    """
    results_dict = {}
    for i, dot_product in enumerate(similarities):
        if dot_product > thresh:
            results_dict[users_to_compare_to[i]] = float("{0:.4f}".format(dot_product))

    return results_dict

def _get_dense_similarities_batch(user_tuple):
    """
        Actual function called by pool workers to calculate the needed dot products per user

        Shoud be used when the created matrix storing each user's entity visitation pattern is a dense matrix

        Calls _find_similarities and formats similarities via _format_similarities

        Params:
            user_tuple (tup) : user_id, list of user_ids to compare user to

        Returns:
            tup : user_id, dict -- key = user_id | value = dot_product
    """
    user_id = user_tuple[0]
    users_to_compare_to = user_tuple[1]

    results = _find_similarities(user_id, USER_ENTITY_MATRIX, users_to_compare_to, sparse=False)

    return (user_id, _format_similarities(users_to_compare_to, results))

def _get_sparse_similarities_batch(user_tuple):
    """
        Actual function called by pool workers to calculate the needed dot products per user

        Shoud be used when the created matrix storing each user's entity visitation pattern is a sparse matrix

        Calls _find_similarities and formats similarities via _format_similarities

        Params:
            user_tuple (tup) : user_id, list of user_ids to compare user to

        Returns:
            tup : user_id, dict -- key = user_id | value = dot_product
    """
    user_id = user_tuple[0]
    users_to_compare_to = user_tuple[1]

    results = _find_similarities(user_id, USER_ENTITY_MATRIX, users_to_compare_to, sparse=True)

    return (user_id, _format_similarities(users_to_compare_to, results))

def prune_space_batch(file_names, n_processes=None, user_cap=DEFAULT_USER_CAP):
    """
        Function that sets up the mulitprocessing environment and sets off the extraction of either the full
        list of possible nearest neighbors or the approximate top_n nearest neighbors for each user.

        Excpets up to three pickle files names:
            1. file name for the user_entity dictionary
            2. file name for the entity_user dictionary
            3. file name for a list of user ids whose similar users are desired
                * if this isn't provided then all users will be used

        To extract the full list of possible nearest neighbors (i.e. no approximation) set user_cap to -1

        Params:
            file_names  (arr) : array of the three files mentioned above
            n_processes (int) : number of processes danny should use when extracting possible
                                nearest neighbors. If left None, danny will use 2 less than the number
                                of cores available on your machine.
            user_cap    (int) : the number of top users that should be extracted in the approximate mode, or
                                to get the full list of possible neighbors pass in -1

        Returns:
            (arr) : each element in the array is a tuple of the following form
                    (user_id, list of relevant user_ids to check for that user)

    """
    # pylint: disable=global-statement, too-many-arguments, too-many-locals
    global USER_ENTITY_DICT, ENTITY_USER_DICT
    start_time = time.time()
    USER_ENTITY_DICT = read_pickle_file(file_names[0])
    ENTITY_USER_DICT = read_pickle_file(file_names[1])

    if len(file_names) == 3:
        users_to_check = read_pickle_file(file_names[2])
    else:
        users_to_check = list(USER_ENTITY_DICT.keys())

    logging.info("read in dictionary pickle files in %s seconds", time.time() - start_time)
    start_time = time.time()
    if user_cap > 0:
        user_indicies = []
        for key in users_to_check:
            user_indicies.append((key, user_cap))
    else:
        user_indicies = users_to_check

    logging.info("prepped users to be analyzed in %s seconds", time.time() - start_time)
    start_time = time.time()

    n_processes = MAX_PROCESSES - 2 if n_processes is None else n_processes

    pool = Pool(processes=n_processes)

    user_tuples = pool.map(_get_top_n_users_batch, user_indicies) if user_cap > 0 \
                  else pool.map(_get_relevant_users_batch, user_indicies)
   
    logging.info("Pruning took %s seconds", time.time() - start_time)
    start_time = time.time()
   
    pool.close()
    pool.join()
    del USER_ENTITY_DICT
    del ENTITY_USER_DICT
    del user_indicies
    del pool
    gc.collect()

    logging.info("Deleting dictionaries took %s seconds", time.time() - start_time)

    return user_tuples

def matrix_multiplication_batch(file_names, user_tuples_list=None, n_processes=None, sparse=True):
    """
        Function that sets up the multiprocessing environment and sets off the calculation of dot products
        for each user.

        Excpets up to two pickle files names:
            1. file name for the user_entity matrix
            2. file name for the user_tuples list genereated by prune_space_batch (or data of similar format)
                * if this isn't provided, the actual list must be passed in

        Params:
            file_names       (arr) : array of the two files mentioned above
            user_tuples_list (arr) : array where each element is a tuple (user_id, list of other user_ids)
                                     the format must match that of the output of prune_space_batch
            n_processes      (int) : number of processes danny should use when extracting possible
                                     nearest neighbors. If left None, danny will use 2 less than the number
                                     of cores available on your machine.
            sparse          (bool) : indicates whether the user_entity_matrix is sparse or not

        Returns:
            dict : key - user_id | value - dict -- key - user_id, value: dot product
    """
    #pylint: disable=global-statement
    global USER_ENTITY_MATRIX
    start_time = time.time()
    USER_ENTITY_MATRIX = read_pickle_file(file_names[0])
    if len(file_names) < 2 and not isinstance(user_tuples_list, list):
        raise ValueError("you must either pass in a file name for the output of prune_space_batch, or \
                          the list it outputs")
   
    user_tuples = read_pickle_file(file_names[1]) if len(file_names) > 1 else user_tuples_list

    logging.info("read in pickle file in %s seconds", time.time() - start_time)
    start_time = time.time()

    n_processes = MAX_PROCESSES - 2 if n_processes is None else n_processes
    pool = Pool(processes=n_processes)

    dictionary_result_tuples = pool.map(_get_sparse_similarities_batch, user_tuples) if sparse \
                               else pool.map(_get_dense_similarities_batch, user_tuples)

    logging.info("Matrix Multiplications took %s seconds", time.time() - start_time)
    start_time = time.time()

    pool.close()
    pool.join()
    del USER_ENTITY_MATRIX
    del user_tuples
    del pool
    gc.collect()

    logging.info("Deleting matrix took %s seconds", time.time() - start_time)
    start_time = time.time()
   
    similarity_scores = {}
    for dict_result_tuple in dictionary_result_tuples:
        user_id = dict_result_tuple[0]
        dict_result = dict_result_tuple[1]
        similarity_scores[user_id] = dict_result

    return similarity_scores

def get_nearest_neighbors_batch(input_type="default", file_names=None, sparse=True, user_cap=DEFAULT_USER_CAP,
                                n_processes=None, save=True, output_dir=DEFAULT_DIR):
    """
        Function that calls prune_space_batch and matrix_multiplication_batch in order to extract per user
        all users who have a dot product above zero or the approximate top_n closest users.

        Expects three pickle files:
            1. file name for the user_entity dictionary, key - user_id | value - dict
                -- key - entity_id, value - number of times user_id visited entity_id
            2. file name for the entity_user dictionary, key - entity_id | value - dict
                -- key - user_id, value - number of times user_id visited entity_id
            3. file name for user_entity matrix, rows represent either one hot or count vectors describing
               the visitation pattern for each user

        These three file names can either be passed in or if using the default file names selected by danny
        just left blank, as danny will know where to find them

        Params:
            input_type  (str) : either "default" or "files" indicating where to find the needed pickle
                                files
            file_names  (arr) : array of the three files mentioned above, can be left blank if using
                                "default" mode
            user_cap    (int) : the number of top users that should be extracted in the approximate mode, or
                                to get the full list of possible neighbors pass in -1 (dot product > 0)
            n_processes (int) : number of processes danny should use when extracting possible
                                nearest neighbors. If left None, danny will use 2 less than the number
                                of cores available on your machine
            save       (bool) : whether to save the output or not
            output_dir  (str) : the directory to write the nearest neighbors per each user to

        Returns:
            bool | dict : if save=True then the function returns True if saving was successful, else
                          it returns the dictionary it would otherwise save. The dictionary is of the
                          following format: key - user_id | value - dict -- key - user_id, value: dot product
    """
    #pylint: disable=too-many-arguments
    input_types = ["default", "files"]
    if input_type not in input_types:
        raise ValueError("input_type must be \"default\" or \"files\"")

    if input_type == "files" and (not isinstance(file_names, list) or len(file_names) < 3):
        raise ValueError("an array of length 3 must be passed in for \"file_names\" indicating \
            the path+filename of the two dictionary files and the matrix file. The order of the \
            files should be: 1. user_entity_dict, 2. entity_user_dict, 3. user_entity_matrix \
            and if needed 4. users_interested_in_array")

    if n_processes is not None and (not isinstance(n_processes, int) or n_processes > MAX_PROCESSES):
        raise ValueError("n_processes must be an int smaller than {}, as your computer only has {} \
            cores".format(MAX_PROCESSES, MAX_PROCESSES))

    user_entity_dict_file_name = file_names[0] if input_type == "files" else \
                                 output_dir + "user_entity_dict.pickle"
    entity_user_dict_file_name = file_names[1] if input_type == "files" else \
                                 output_dir + "entity_user_dict.pickle"
    user_entity_matrix_file_name = file_names[2] if input_type == "files" else \
                                   output_dir + "user_entity_matrix.pickle"

    dict_file_names = [user_entity_dict_file_name, entity_user_dict_file_name]
  
    if len(file_names) == 4:
        dict_file_names.append(file_names[3])

    user_tuples = prune_space_batch(dict_file_names, n_processes, user_cap)
    gc.collect()

    similarity_scores = matrix_multiplication_batch(user_entity_matrix_file_name,
                                                    user_tuples,
                                                    n_processes,
                                                    sparse)
    del user_tuples
    gc.collect()

    if save:
        similarity_scores_file_name = output_dir + "similarity_scores.pickle"
        write_pickle_file(similarity_scores, similarity_scores_file_name)

        del similarity_scores
        return True
  
    return similarity_scores
