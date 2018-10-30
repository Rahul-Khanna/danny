import pickle
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize

DEFAULT_DIR = "output_data/"

def read_pickle_file(file_name):
    # pylint: disable=invalid-name
    with open(file_name, "rb") as f:
        data = pickle.load(f)
    
    return data

def write_pickle_file(data, file_name):
    # pylint: disable=invalid-name
    with open(file_name, "wb") as f:
        pickle.dump(data, f)

    return data

def create_dictionaries(raw_log_file, one_hot=False, save=True, output_dir=DEFAULT_DIR):
    # pylint: disable=invalid-name, unused-variable
    with open(raw_log_file) as f:
        logs = pd.read_csv(f, header=None, names=["user_id", "entity_id"])

    user_entity_dict = {}
    entity_user_dict = {}

    for i, row in logs.iterrows():
        user_id = row["user_id"]
        entity_id = row["entity_id"]

        if user_id in user_entity_dict:
            if entity_id in user_entity_dict[user_id] and not one_hot:
                user_entity_dict[user_id][entity_id] += 1
            else:
                user_entity_dict[user_id][entity_id] = 1
        else:
            user_entity_dict[user_id] = {}
            user_entity_dict[user_id][entity_id] = 1

        if entity_id in entity_user_dict:
            if user_id in entity_user_dict[entity_id] and not one_hot:
                entity_user_dict[entity_id][user_id] += 1
            else:
                entity_user_dict[entity_id][user_id] = 1
        else:
            entity_user_dict[entity_id] = {}
            entity_user_dict[entity_id][user_id] = 1

    if save:
        user_entity_dict_file_name = output_dir + "user_entity_dict.pickle"
        entity_user_dict_file_name = output_dir + "entity_user_dict.pickle"

        write_pickle_file(user_entity_dict, user_entity_dict_file_name)
        write_pickle_file(entity_user_dict, entity_user_dict_file_name)

        del user_entity_dict
        del entity_user_dict
        return True

    return (user_entity_dict, entity_user_dict)

def create_matrix(input_type="default", data_source=None, sparse=True, save=True, output_dir=DEFAULT_DIR):

    input_types = ["default", "file", "dict"]
    
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

    user_dicts = list(data_source.values())

    vectorizer = DictVectorizer(sparse=sparse)
    user_entity_matrix = vectorizer.fit_transform(user_dicts)
    user_entity_matrix = normalize(user_entity_matrix)

    if save:
        user_entity_matrix_file_name = output_dir + "user_entity_matrix.pickle"
        write_pickle_file(user_entity_matrix, user_entity_matrix_file_name)

        del user_entity_matrix
        return True
    
    return user_entity_matrix
