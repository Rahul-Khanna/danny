import argparse
import logging
import supporting_functions
import dictionary_based_nn


def main():
    #pylint: disable=too-many-branches, too-many-statements
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dictionary", "matrix", "nn", "index", "batch"], const="index",
                        nargs='?', help="what operations should danny perform")
    parser.add_argument("--log_file", nargs='?', help="csv containing logs to be processed")
    parser.add_argument("--user_entity_dict_file", nargs='?', help="pickle file holding \
                        user_entity_dictionary")
    parser.add_argument("--entity_user_dict_file", nargs='?', help="pickle file holding \
                        entity_user_dictionary")
    parser.add_argument("--user_entity_matrix_file", nargs='?', help="pickle file holding \
                         entity_user_matrix")
    parser.add_argument("--users_to_check_file", nargs='?', help="users whose similarities are required")
    parser.add_argument("--dense", action="store_true", help="the user_entity matrix will be dense or not")
    parser.add_argument("--user_cap", type=int, nargs='?', help="cap on how many user similarity scores \
                        should be calculated, if -1 then no cap is used")
    parser.add_argument("--processes", type=int, nargs='?', help="number of processes to use when finding \
                        nearest neighbors")
    parser.add_argument("--one_hot", action="store_true", help="where all files should be \
                        outputted to")
    parser.add_argument("--output_dir", nargs='?', help="where all files should be outputted to")
    parser.add_argument("--verbose", action="store_true", help="print out timings for each step of the \
                        process")
    
    args = parser.parse_args()

    user_cap = args.user_cap if args.user_cap else 500
    sparse = False if args.dense else True
    processes = args.processes if args.processes else None
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    if args.mode == "dictionary":
        if args.log_file:
            if args.output_dir:
                supporting_functions.create_dictionaries(args.log_file,
                                                         one_hot=args.one_hot,
                                                         n_processes=processes,
                                                         output_dir=args.output_dir)
                print("saved dictionaries to {}".format(args.output_dir))
            else:
                supporting_functions.create_dictionaries(args.log_file,
                                                         one_hot=args.one_hot,
                                                         n_processes=processes)
                print("saved dictionaries to \"output_data\"")
        else:
            raise ValueError("need log file to convert into dictionaries")

    if args.mode == "matrix":
        if args.user_entity_dict_file:
            if args.output_dir:
                supporting_functions.create_matrix(input_type="file",
                                                   data_source=args.user_entity_dict_file,
                                                   sparse=sparse,
                                                   output_dir=args.output_dir)
                print("saved matrix to {}".format(args.output_dir))
            else:
                supporting_functions.create_matrix(input_type="file",
                                                   data_source=args.user_entity_dict_file,
                                                   sparse=sparse)
                print("saved matrix to \"output_data\"")

        else:
            if args.output_dir:
                supporting_functions.create_matrix(sparse=sparse, output_dir=args.output_dir)
                print("saved matrix to {}".format(args.output_dir))
            else:
                supporting_functions.create_matrix(sparse=sparse)
                print("saved matrix to \"output_data\"")

    if args.mode == "nn":
        if args.user_entity_dict_file and args.entity_user_dict_file and args.user_entity_matrix_file:
            file_1 = args.user_entity_dict_file
            file_2 = args.entity_user_dict_file
            file_3 = args.user_entity_matrix_file
            
            if args.users_to_check_file:
                file_names = [file_1, file_2, file_3, args.users_to_check_file]
            else:
                file_names = [file_1, file_2, file_3]

            if args.output_dir:
                dictionary_based_nn.get_nearest_neighbors_batch(input_type="files",
                                                                file_names=file_names,
                                                                sparse=sparse,
                                                                user_cap=user_cap,
                                                                n_processes=processes,
                                                                output_dir=args.output_dir)
                print("saved similarity scores to {}".format(args.output_dir))
            else:
                dictionary_based_nn.get_nearest_neighbors_batch(input_type="files",
                                                                file_names=file_names,
                                                                sparse=sparse,
                                                                user_cap=user_cap,
                                                                n_processes=processes)
                print("saved similarity scores to \"output_data\"")
        else:
            if args.output_dir:
                dictionary_based_nn.get_nearest_neighbors_batch(sparse=sparse,
                                                                user_cap=user_cap,
                                                                n_processes=processes,
                                                                output_dir=args.output_dir)
                print("saved similarity scores to {}".format(args.output_dir))
            else:
                dictionary_based_nn.get_nearest_neighbors_batch(sparse=sparse,
                                                                user_cap=user_cap,
                                                                n_processes=processes)
                print("saved similarity scores to \"output_data\"")

    if args.mode == "index":
        if args.log_file:
            if args.output_dir:
                supporting_functions.create_dictionaries(args.log_file,
                                                         one_hot=args.one_hot,
                                                         n_processes=processes,
                                                         output_dir=args.output_dir)
                print("saved dictionaries to {}".format(args.output_dir))
            else:
                supporting_functions.create_dictionaries(args.log_file,
                                                         one_hot=args.one_hot,
                                                         n_processes=processes)
                print("saved dictionaries to \"output_data\"")
        else:
            raise ValueError("need log file to convert into dictionaries")

        if args.output_dir:
            supporting_functions.create_matrix(sparse=sparse, output_dir=args.output_dir)
            print("saved matrix to {}".format(args.output_dir))
        else:
            supporting_functions.create_matrix(sparse=sparse)
            print("saved matrix to \"output_data\"")

    if args.mode == "batch":
        if args.log_file:
            if args.output_dir:
                supporting_functions.create_dictionaries(args.log_file,
                                                         one_hot=args.one_hot,
                                                         n_processes=processes,
                                                         output_dir=args.output_dir)
                print("saved dictionaries to {}".format(args.output_dir))
            else:
                supporting_functions.create_dictionaries(args.log_file,
                                                         one_hot=args.one_hot,
                                                         n_processes=processes)
                print("saved dictionaries to \"output_data\"")
        else:
            raise ValueError("need log file to convert into dictionaries")

        if args.output_dir:
            supporting_functions.create_matrix(sparse=sparse, output_dir=args.output_dir)
            print("saved matrix to {}".format(args.output_dir))
        else:
            supporting_functions.create_matrix(sparse=sparse)
            print("saved matrix to \"output_data\"")

        if args.output_dir:
            dictionary_based_nn.get_nearest_neighbors_batch(sparse=sparse,
                                                            user_cap=user_cap,
                                                            n_processes=processes,
                                                            output_dir=args.output_dir)
            print("saved similarity scores to {}".format(args.output_dir))
        else:
            dictionary_based_nn.get_nearest_neighbors_batch(sparse=sparse,
                                                            user_cap=user_cap,
                                                            n_processes=processes)
            print("saved similarity scores to \"output_data\"")

if __name__ == '__main__':
    main()
