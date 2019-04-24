"""
    Wrapper script for danny. Allows a user to interact with danny's functionality via the command line by
    passing in arugments.
        Ex: python dannyw.py --mode=build_index, --log_file=data/raw_logs.csv --one_hot --verbose
            ^ - will build the three needed data structures for the nearest neighbors search from the logs
                stored in "data". Will encode user entity visitation patterns in a one_hot encoded way, not
                as a count vector. Finally will print out helpful statements about timing and where danny
                is in the pipeline

    You can interact with the following functionality through this wrapper script:
        1. re_index    - ensures your user and entity ids are consecutive ints starting from zero
        2. dictionary  - builds the needed user_entity_dictionary and entity_user_dictionary from a properly
                         formatted log file
        3. matrix      - builds the needed user_entity_matrix from the user_entity_dictionary
        4. nn          - by default will compute the nearest neighbors for all users in approximate mode.
                         The number of nearest neighbors can either be a positive int below 1000, or -1 --
                         indicates no cap and to use the smart, but comprehensive mode of danny
        5. build_index - builds all three of the needed data structures for danny to figure out nearest
                         neighbors from a properly formatted log file. Essentially runs the "dictionary" and
                         then "matrix" option.
        6. batch       - computes nearest neighbors for each user from a properly formatted log file.
                         Essentially runs the "dictionary", "matrix" and finally "nn" options. Important to
                         read how to configure the "nn" to your liking.

    danny will take care of the file storage for you if you want. It will save all data in a folder called
    "output_data", so make sure that exists in the directory you are running this script from. If you have
    already run setup.sh then you are fine.

    You can also pass in your own output directory if you want

    For more information on the other command line flags/arguments run, python dannyw.py --help
"""
import argparse
import logging
import supporting_functions
import dictionary_based_nn

def main():
    #pylint: disable=too-many-branches, too-many-statements, missing-docstring
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["re_index", "dictionary", "matrix", "nn", "build_index", "batch"],
                        const="index", nargs='?', help="what operation should danny perform")
    parser.add_argument("--log_file", nargs='?', help="csv containing logs to be processed")
    parser.add_argument("--user_entity_dict_file", nargs='?', help="pickle file holding \
                        user_entity_dictionary")
    parser.add_argument("--entity_user_dict_file", nargs='?', help="pickle file holding \
                        entity_user_dictionary")
    parser.add_argument("--user_entity_matrix_file", nargs='?', help="pickle file holding \
                         entity_user_matrix")
    parser.add_argument("--users_to_check_file", nargs='?', help="users whose similarities are required")
    parser.add_argument("--dense", action="store_true", help="the user_entity matrix should be dense or not")
    parser.add_argument("--user_cap", type=int, nargs='?', help="cap on how many user similarity scores \
                        should be calculated, if -1 then no cap is used")
    parser.add_argument("--processes", type=int, nargs='?', help="number of processes to use when finding \
                        nearest neighbors")
    parser.add_argument("--one_hot", action="store_true", help="should the matrix be constructed from count \
                        vectors or one-hot encoded vectors")
    parser.add_argument("--output_dir", nargs='?', help="where all files should be outputted to")
    parser.add_argument("--verbose", action="store_true", help="print out timings for each step of the \
                        process")
    
    args = parser.parse_args()

    user_cap = args.user_cap if args.user_cap else 500
    sparse = False if args.dense else True
    processes = args.processes if args.processes else None
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    if args.mode == "re_index":
        if args.log_file:
            if args.output_dir:
                supporting_functions.reindex_log_file(args.log_file, output_dir=args.output_dir)
                print("saved re-indexed log file and one way mapping to {}".format(args.output_dir))
            else:
                supporting_functions.reindex_log_file(args.log_file)
                print("saved re-indexed log file and one way mapping to \"output_data\"")
        else:
            raise ValueError("need log file to re-index")


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

    if args.mode == "build_index":
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
