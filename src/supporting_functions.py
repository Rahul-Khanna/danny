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
    # pylint: disable=invalid-name
    with open(file_name, "rb") as f:
        data = pickle.load(f)
    
    return data

def write_pickle_file(data, file_name):
    # pylint: disable=invalid-name
    with open(file_name, "wb") as f:
        pickle.dump(data, f)

    return True

def _create_count_mini_dictionaries(logs):
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

def reindex_log_file(raw_log_file, save=True, output_dir=DEFAULT_DIR):
    user_index = {}
    entity_index = {}
    user_count = 0
    entity_count = 0
    new_logs = []
    start_time.time()
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

        logging.info("wrote out indicies in %s seconds", time.time() - start_time)
        start_time = time.time()

        del user_index
        del entity_index
        del new_logs
        return True
    
    return (new_logs, user_index, entity_index)

def reverse_index(input_type, data_source, index_type=None, save=True, output_dir=DEFAULT_DIR):
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


def create_matrix(input_type="default", data_source=None, sparse=True, save=True, output_dir=DEFAULT_DIR):

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
