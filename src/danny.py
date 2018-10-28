import argparse
import supporting_functions
import dictionary_based_nn


def main():
    #pylint: disable=bad-continuation, too-many-branches, too-many-statements
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dictionary", "matrix", "ann", "all"], const="all",
                        help="what stage of the pipeline needs to be run")
    parser.add_argument("--log_file", help="csv containing logs to be processed")
    parser.add_argument("--user_entity_dict_file", help="pickle file holding user_entity_dictionary")
    parser.add_argument("--entity_user_dict_file", help="pickle file holding entity_user_dictionary")
    parser.add_argument("--user_entity_matrix_file", help="pickle file holding entity_user_matrix")
    parser.add_argument("--dense", action="store_true", help="the user_entity matrix will be / is dense")
    parser.add_argument("--user_cap", type="int", help="cap on how many user similarity scores should be \
                        calculated, if -1 then no cap is used")
    parser.add_argument("--processes", type="int", help="number of processes to use when finding nearest \
                        neighbors")
    parser.add_argument("--output_dir", help="where all files should be outputted to")
    
    args = parser.parse_args()

    user_cap = args.user_cap if args.user_cap else 500
    sparse = False if args.dense else True
    processes = args.processes if args.processes else None

    if args.mode == "dictionary":
        if args.log_file:
            if args.output_dir:
                supporting_functions.create_dictionaries(args.log_file, output_dir=args.output_dir)
                print("saved dictionaries to {}".format(args.output_dir))
            else:
                supporting_functions.create_dictionaries(args.log_file)
                print("saved dictionaries to \"output_data\"")
        else:
            raise ValueError("need log file to convert into dictionaries")

    if args.mode == "matrix":
        if args.user_entity_dict_file:
            if args.output_dir:
                supporting_functions.create_matrix(input_type="file",
                                                   data_source=args.user_entity_dict_file,
                                                   output_dir=args.output_dir,
                                                   sparse=sparse)
                print("saved matrix to {}".format(args.output_dir))
            else:
                supporting_functions.create_matrix(input_type="file",
                                                   data_source=args.user_entity_dict_file,
                                                   sparse=sparse)
                print("saved matrix to \"output_data\"")

        else:
            if args.output_dir:
                supporting_functions.create_matrix(output_dir=args.output_dir, sparse=sparse)
                print("saved matrix to {}".format(args.output_dir))
            else:
                supporting_functions.create_matrix(sparse=sparse)
                print("saved matrix to \"output_data\"")

    if args.mode == "ann":
        if args.user_entity_dict_file and args.entity_user_dict_file and args.user_entity_matrix_file:
            if args.output_dir:
                dictionary_based_nn.get_nearest_neighbors(input_type="files",
                    file_names=[args.user_entity_dict_file, args.entity_user_dict_file, \
                    args.user_entity_matrix_file],
                    sparse=sparse, output_dir=args.output_dir, user_cap=user_cap, n_processes=processes)
                print("saved similarity scores to {}".format(args.output_dir))
            else:
                dictionary_based_nn.get_nearest_neighbors(input_type="files",
                    file_names=[args.user_entity_dict_file, args.entity_user_dict_file, \
                    args.user_entity_matrix_file],
                    sparse=sparse, user_cap=user_cap, n_processes=processes)
                print("saved similarity scores to \"output_data\"")
        else:
            if args.output_dir:
                dictionary_based_nn.get_nearest_neighbors(sparse=sparse, output_dir=args.output_dir,
                                                          user_cap=user_cap, n_processes=processes)
                print("saved similarity scores to {}".format(args.output_dir))
            else:
                dictionary_based_nn.get_nearest_neighbors(sparse=sparse, user_cap=user_cap,
                                                          n_processes=processes)
                print("saved similarity scores to \"output_data\"")

    if args.mode == "all":
        if args.log_file:
            if args.output_dir:
                supporting_functions.create_dictionaries(args.log_file, output_dir=args.output_dir)
                print("saved dictionaries to {}".format(args.output_dir))
            else:
                supporting_functions.create_dictionaries(args.log_file)
                print("saved dictionaries to \"output_data\"")
        else:
            raise ValueError("need log file to convert into dictionaries")

        if args.output_dir:
            supporting_functions.create_matrix(output_dir=args.output_dir, sparse=sparse)
            print("saved matrix to {}".format(args.output_dir))
        else:
            supporting_functions.create_matrix(sparse=sparse)
            print("saved matrix to \"output_data\"")

        if args.output_dir:
            dictionary_based_nn.get_nearest_neighbors(sparse=sparse, output_dir=args.output_dir,
                                                      user_cap=user_cap, n_processes=processes)
            print("saved similarity scores to {}".format(args.output_dir))
        else:
            dictionary_based_nn.get_nearest_neighbors(sparse=sparse, user_cap=user_cap,
                                                      n_processes=processes)
            print("saved similarity scores to \"output_data\"")

if __name__ == '__main__':
    main()
