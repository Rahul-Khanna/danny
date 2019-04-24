"""
    Functions that support the actual finding of nearest neighbors per user. The functions can be grouped
    by the following:
        1. Interacting with pickle files
        2. Ensuring the user_ids and entity_ids passed in are consecutive integers starting from 0
        3. Building count or one_hot dictionaries describing user visitation patterns / entity visitation
           patterns
        4. Building a user_entity_matrix

    These functions ensure the data being fed into danny is as expected, and then creates the three needed
    data structures danny needs to operate:
        1. user_entity_dict: does so in a parrallel way
        2. entity_user_dict: does so in a parrallel way
        3. user_entity_dict: uses scikit learn

    You can think of these functions as building danny's index so that danny can later query who the close
    users are for each user.

    Important Functions:
        1. reindex_log_file
        2. create_dictionaries
        3. create_matrix
"""
import logging
from multiprocessing import Pool, cpu_count
from operator import itemgetter
import pickle
import time
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize

DEFAULT_DIR = "output_data/"
MAX_PROCESSES = cpu_count()
MAX_LOG_CHUNK = 500000

def read_pickle_file(file_name):
    """
        Reads a pickle file

        Params:
            file_name (str) : name of file to read

        Returns:
            Object : whatever data was pickled
    """
    # pylint: disable=invalid-name
    with open(file_name, "rb") as f:
        data = pickle.load(f)
    
    return data

def write_pickle_file(data, file_name):
    """
        Writes a pickle file

        Params:
            data   (Object) : the data that needs to be pickled
            file_name (str) : name of file to read

        Returns:
            bool : True on completion
    """
    # pylint: disable=invalid-name
    with open(file_name, "wb") as f:
        pickle.dump(data, f)

    return True

def reindex_log_file(raw_log_file, save=True, output_dir=DEFAULT_DIR):
    """
        Function reads a log file of the expected format of: user_id, entity_id and reindexes users and
        entities to ensure that user_ids and entity_ids start from zero and are consecutive. The result
        of the conversion and the mapping between the old indicies and the new ones can either be returned
        or written out. The log file will be writeen out in the expected log format, while the mappings will
        be written out as pickle files.

        In order to preserve links between a user and their entity visitation pattern, or an entity and its
        user visitation pattern, user and entity ids must start at 0 and be consecutive. For more on this you
        can read the readme.

        Params:
            raw_log_file (str) : name of log file to reindex
            save        (bool) : boolean to indicate whether to save the results of reindexing
            output_dir   (str) : directory to write out to

        Returns:
            tup | bool : if the results are not to be saved the function returns:
                         (array of reindexed logs, user index mapping, entity index mapping)
                         else it returns True to indicate the results were saved
    """
    #pylint: disable=too-many-locals, invalid-name
    user_index = {}
    entity_index = {}
    user_count = 0
    entity_count = 0
    new_logs = []
    start_time = time.time()
    with open(raw_log_file) as logs:
        for line in logs:
            parts = line.split(",")
            user_id = int(parts[0])
            entity_id = int(parts[1])

            if user_id not in user_index:
                user_index[user_id] = user_count
                user_count += 1
            
            if entity_id not in entity_index:
                entity_index[entity_id] = entity_count
                entity_count += 1

            new_logs.append(str(user_index[user_id]) + "," + str(entity_index[entity_id]))

    logging.info("read in and converted logs in %s seconds", time.time() - start_time)
    start_time = time.time()

    if save:
        converted_log_file_name = output_dir + "converted_logs.csv"
        with open(converted_log_file_name, "w") as f:
            for i, line in enumerate(new_logs):
                if i < len(new_logs) -1:
                    f.write(line+"\n")
                else:
                    f.write(line)
        
        logging.info("wrote out converted logs in %s seconds", time.time() - start_time)
        start_time = time.time()

        user_index_file = output_dir + "user_index.pickle"
        entity_index_file = output_dir + "entity_index.pickle"
        write_pickle_file(user_index, user_index_file)
        write_pickle_file(entity_index, entity_index_file)

        logging.info("wrote out mappings in %s seconds", time.time() - start_time)
        start_time = time.time()

        del user_index
        del entity_index
        del new_logs
        return True
    
    return (new_logs, user_index, entity_index)

def reverse_index(input_type, data_source, index_type=None, save=True, output_dir=DEFAULT_DIR):
    """
        Function swithces around key value pairs of a dictionary to allow for both forward and backward
        switching between new and old indicies for users and entities.

        When inspecting the results of the nearest neighbors it is useful to have the reverse index to see
        which original user_ids are close to each other.

        Params:
            input_type  (str) : indicates whether a file or dict is being passed into the function
            data_source (str) : the source of data, so either the file or dict
            index_type  (str) : indicating what type of index is being saved, i.e. user_index
                                can be left None if output is not being saved
            save       (bool) : whether to save the output of the function
            output_dir  (str) : directory to write out to

        Returns:
            tup | bool : if the results are not to be saved the function returns the reversed_index
                         else it returns True to indicate the results were saved
    """
    input_types = ["file", "dict"]
    if input_type not in input_types:
        raise ValueError("input_type must be one of \"file\" or \"dict\"")

    if input_type == "file" and not isinstance(data_source, str):
        raise ValueError("data_source must indicated the pickle file you would like to be read in to \
            to reverse your index")

    if input_type == "dict" and not isinstance(data_source, dict):
        raise ValueError("data_source must be the needed dictionary/index")

    if save and index_type is None:
        raise ValueError("index_type cannot be null if the reveresed index is going to be saved. Common \
            options are: entity, user, site, song, video")

    reversed_index = {}

    if input_type == "file":
        index = read_pickle_file(data_source)
    else:
        index = data_source

    for key in index:
        reversed_index[index[key]] = key

    if save:
        reversed_index_file_name = output_dir + index_type + "_reverse_index.pickle"
        write_pickle_file(reversed_index, reversed_index_file_name)

        del reversed_index
        return True

    return reversed_index

def _create_count_mini_dictionaries(logs):
    """
        Function called by pool workers to parralelize the process of creating two count dictionaries from
        the expected log file format. The two dictionaries are the user-entity dictionary and the entity-user
        dictionary, and are of following format:
        
        user-entity dict:
            key - user_id
            value - dict :
                           key - entity_id
                           value - count of the number time user_id visited entity_id

        entity-user dict:
            key - entity_id
            value - dict :
                           key - user_id
                           value - count of the number time user_id visited entity_id

        The original log file is split into mulitple parts, each for a pool worker to consume, create their
        version of these dictionaries that will then be merged at the end to create the comprehensive
        dictionaries

        Params:
            logs (arr) : array of strings of the following format: user_id, entity_id

        Returns:
            tup : user_entity_dict, entity_user_dict
    """
    user_entity_dict = {}
    entity_user_dict = {}

    for line in logs:
        parts = line.rstrip().split(",")
        user_id = int(parts[0])
        entity_id = int(parts[1])

        if user_id not in user_entity_dict:
            user_entity_dict[user_id] = {}

        if entity_id in user_entity_dict[user_id]:
            user_entity_dict[user_id][entity_id] += 1
        else:
            user_entity_dict[user_id][entity_id] = 1

        if entity_id not in entity_user_dict:
            entity_user_dict[entity_id] = {}

        if user_id in entity_user_dict[entity_id]:
            entity_user_dict[entity_id][user_id] += 1
        else:
            entity_user_dict[entity_id][user_id] = 1

    return (user_entity_dict, entity_user_dict)

def _create_one_hot_mini_dictionaries(logs):
    """
        Function called by pool workers to parralelize the process of creating two one hot dictionaries from
        the expected log file format. The two dictionaries are the user-entity dictionary and the entity-user
        dictionary, and are of following format:
        
        user-entity dict:
            key - user_id
            value - dict :
                           key - entity_id
                           value - 1

        entity-user dict:
            key - entity_id
            value - dict :
                           key - user_id
                           value - 1

        The original log file is split into mulitple parts, each for a pool worker to consume, create their
        version of these dictionaries that will then be merged at the end to create the comprehensive
        dictionaries

        Params:
            logs (arr) : array of strings of the following format: user_id, entity_id

        Returns:
            tup : user_entity_dict, entity_user_dict
    """

    user_entity_dict = {}
    entity_user_dict = {}

    for line in logs:
        parts = line.rstrip().split(",")
        user_id = int(parts[0])
        entity_id = int(parts[1])

        if user_id not in user_entity_dict:
            user_entity_dict[user_id] = {}

        if entity_id not in user_entity_dict[user_id]:
            user_entity_dict[user_id][entity_id] = 1

        if entity_id not in entity_user_dict:
            entity_user_dict[entity_id] = {}

        if user_id not in entity_user_dict[entity_id]:
            entity_user_dict[entity_id][user_id] = 1

    return (user_entity_dict, entity_user_dict)

def _combine_count_mini_dictionaries(mini_dicionaries):
    """
        Combines the partially completed count dictionaries created by the pool workers in order to have
        two comprehensive dictionaries, the user_entity_dict and entity_user_dict.

        Params:
            mini_dicionaries (arr) : each element is a tuple containing:
                                     (user_entity_mini_dict, entity_user_mini_dict)

        Returns:
            tup : comprehensive_user_entity_dict, comprehensive_entity_user_dict
    """
    user_entity_dict = {}
    entity_user_dict = {}

    for pair in mini_dicionaries:
        user_mini_dict = pair[0]
        entity_mini_dict = pair[1]
        
        for user in user_mini_dict:
            if user not in user_entity_dict:
                user_entity_dict[user] = {}

            for entity in user_mini_dict[user]:
                if entity in user_entity_dict[user]:
                    user_entity_dict[user][entity] += user_mini_dict[user][entity]
                else:
                    user_entity_dict[user][entity] = user_mini_dict[user][entity]

        for entity in entity_mini_dict:
            if entity not in entity_user_dict:
                entity_user_dict[entity] = {}

            for user in entity_mini_dict[entity]:
                if user in entity_user_dict[entity]:
                    entity_user_dict[entity][user] += entity_mini_dict[entity][user]
                else:
                    entity_user_dict[entity][user] = entity_mini_dict[entity][user]

    return(user_entity_dict, entity_user_dict)

def _combine_one_hot_mini_dictionaries(mini_dicionaries):
    """
        Combines the partially completed one hot dictionaries created by the pool workers in order to have
        two comprehensive dictionaries, the user_entity_dict and entity_user_dict.

        Params:
            mini_dicionaries (arr) : each element is a tuple containing:
                                     (user_entity_mini_dict, entity_user_mini_dict)

        Returns:
            tup : comprehensive_user_entity_dict, comprehensive_entity_user_dict
    """
    user_entity_dict = {}
    entity_user_dict = {}
    for pair in mini_dicionaries:
        user_mini_dict = pair[0]
        entity_mini_dict = pair[1]
        
        for user in user_mini_dict:
            if user not in user_entity_dict:
                user_entity_dict[user] = {}

            for entity in user_mini_dict[user]:
                if entity not in user_entity_dict[user]:
                    user_entity_dict[user][entity] = 1

        for entity in entity_mini_dict:
            if entity not in entity_user_dict:
                entity_user_dict[entity] = {}
            
            for user in entity_mini_dict[entity]:
                if user not in entity_user_dict[entity]:
                    entity_user_dict[entity][user] = 1

    return (user_entity_dict, entity_user_dict)

def create_dictionaries(raw_log_file, one_hot=False, n_processes=None, save=True,
                        output_dir=DEFAULT_DIR):
    """
        Chunks the raw logs (user_id, entity_id) into units of 500000 lines, sets up a pool of workers, and
        distributes the work of building larger count or one hot dictionaries of the following forms:

        user-entity dict:
            key - user_id
            value - dict :
                           key - entity_id
                           value - count or 1

        entity-user dict:
            key - entity_id
            value - dict :
                           key - user_id
                           value - count or 1

        Each worker creates their own version of these dictionaries, which then get merged into one large
        comprehensive dictionary, which can then be saved or returned to the user.

        Note : for usage in danny, the users and entities in the raw log file must be indexed by consecutive
               numbers starting for zero.

        Params:
            raw_log_file (str) : name of log file to build dictionaries out of
            one_hot     (bool) : a 1 insted of the true count will be used when building the dictionary
                                 use this if you want the resulting user-entity matrix to be one hot encoded
            n_processes  (int) : number of processes danny should use when extracting possible
                                 nearest neighbors. If left None, danny will use 2 less than the number
                                 of cores available on your machine
            save        (bool) : whether to save the output or not
            output_dir   (str) : the directory to write the nearest neighbors per each user to

        Returns:
            tup | bool : if the results are not to be saved the function returns:
                         (user_entity_dict, entity_user_dict)
                         else it returns True to indicate the dictionaries were saved
    """
    # pylint: disable=too-many-arguments, too-many-locals
    n_processes = MAX_PROCESSES - 2 if n_processes is None else n_processes
    chunked_logs = []
    start_time = time.time()
    with open(raw_log_file) as logs:
        i = 0
        chunk = []
        for line in logs:
            if i < MAX_LOG_CHUNK:
                chunk.append(line)
                i += 1
            else:
                chunked_logs.append(chunk)
                chunk = [line]
                i = 1

        chunked_logs.append(chunk)

    logging.info("read in logs in %s seconds", time.time() - start_time)
    start_time = time.time()
    pool = Pool(processes=n_processes)

    mini_dicionaries = pool.map(_create_one_hot_mini_dictionaries, chunked_logs) if one_hot \
                       else pool.map(_create_count_mini_dictionaries, chunked_logs)

    pool.close()
    pool.join()

    logging.info("mini dictionaries created in %s seconds", time.time() - start_time)
    start_time = time.time()
   
    if one_hot:
        combined_dicts = _combine_one_hot_mini_dictionaries(mini_dicionaries)
    else:
        combined_dicts = _combine_count_mini_dictionaries(mini_dicionaries)

    logging.info("mini dictionaries combined in %s seconds", time.time() - start_time)

    if save:
        user_entity_dict_file_name = output_dir + "user_entity_dict.pickle"
        entity_user_dict_file_name = output_dir + "entity_user_dict.pickle"

        write_pickle_file(combined_dicts[0], user_entity_dict_file_name)
        write_pickle_file(combined_dicts[1], entity_user_dict_file_name)

        del combined_dicts
        return True

    return combined_dicts


def create_matrix(input_type="default", data_source=None, sparse=True, save=True, output_dir=DEFAULT_DIR):
    """
        Creates either a one_hot or count matrix, encoding the users' entity visitation patterns in the rows, 
        and each entities' user visitation history in the columns. Takes in a the user_entity_dict and
        creates the needed matrix via sklearn's DictVectorizer function. The matrix can either be sparse or
        dense, with the default being sparse.

        As stated, the function expects the user_entity_dict outputted by create_dictionaries (or data of a
        similar format) to be passed in. This dictionary can either be passed in, read in from a passed in
        file or in the "default" case danny will know where to find the file.

        Params:
            input_type            (str) : how the user_entity_dict is being passed in
            data_source (str|dict|None) : a file name, the user_entity_dict or None in which case danyy will
                                          read in the user_entity_dict from the default location
            sparse               (bool) : whether the user_entity_matrix should be sparse or not, default is
                                          sparse
            save                 (bool) : whether to save the output or not
            output_dir            (str) : the directory to write the nearest neighbors per each user to

        Returns:
            tup | bool : if the results are not to be saved the function returns the user_entity_matrix
                         else it returns True to indicate the results were saved
    """

    input_types = ["default", "file", "dict"]
    start_time = time.time()
    if input_type not in input_types:
        raise ValueError("input_type must be one of \"default\", \"file\" or \"dict\"")

    if input_type == "file" and not isinstance(data_source, str):
        raise ValueError("data_source must indicated the pickle file you would like to be read in to\
            to create the user_entity matrix")

    if input_type == "dict" and not isinstance(data_source, dict):
        raise ValueError("data_source must be the needed dictionary to create the user_entity matrix")

    if input_type in ["file", "default"]:
        file_name = data_source if input_type == "file" else DEFAULT_DIR + "user_entity_dict.pickle"
        data_source = read_pickle_file(file_name)
        logging.info("read in needed pickle files in %s seconds", time.time() - start_time)

    start_time = time.time()
    user_dicts = sorted(data_source.items(), key=itemgetter(0))
    user_dicts = [tup[1] for tup in user_dicts]
   
    logging.info("prepped user info for matrix creation %s seconds", time.time() - start_time)
    start_time = time.time()

    vectorizer = DictVectorizer(sparse=sparse)
    user_entity_matrix = vectorizer.fit_transform(user_dicts)
    logging.info("matrix is created in %s seconds", time.time() - start_time)
    start_time = time.time()

    user_entity_matrix = normalize(user_entity_matrix)
    logging.info("matrix is row normalized in %s seconds", time.time() - start_time)

    if save:
        user_entity_matrix_file_name = output_dir + "user_entity_matrix.pickle"
        write_pickle_file(user_entity_matrix, user_entity_matrix_file_name)

        del user_entity_matrix
        return True
   
    return user_entity_matrix
