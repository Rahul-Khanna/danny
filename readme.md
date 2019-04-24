## D(ictionary-based) A(pproximate) N(earest) (N)eighbor Y(up)

**danny**: A dictionary based approach to the nearest neighbor problem.

### Contents:
* [Quick Summary](#quick-summary)
* [How To Use](#how-to-use)
* [Important File Descriptions](#important-file-descriptions)
* [ETL Pipeline Description](#etl-pipeline-description)
* [Why Build danny](#why-build-danny)
* [Some Theory](#some-theory)
* [Work Still Left To Do](#work-still-left-to-do)
* [Copyright](#copyright)

## Quick Summary
Given a log file of the following format:

    entity_a_id, entity_b_id
describing some sort of consumption, visitation or viewing (etc.) pattern of *entity_a*, **danny will find the approximate top-n nearest neighbors for each entity_a based on each user's pattern.**

For the purpose of **clarity** I will be referring to:

* **entity_a_id** as **user_id** 
* **entity_b_id** as **entity_id**
* the log file as a file describing each user's **entity visitation pattern**

The nearest neighbors per user are then the users who visit the same entities at the same rate (or just the same entities -- both definitions are supported).

As mentioned above, **danny** will use some approximation techniques to find the top-n nearest neighbors per users, however there is an **alternative exact technique** supported by **danny**--in certain conditions the exact technique is more appropriate to use.

Both the approximate method and the exact method are based around the concept of **using dictionaries to reduce the space of users to look at when considering who the nearest neighbors of user_i are**. By using both a *user_entity* and *entity_user* dictionary (more on these dictionaries can be found below), the search space can be reduced per user before having to compute dot products and sort users. **danny** still ultimately computes dot products, but does so for far fewer users than the naive nearest neighbors would.

**danny** also does most of its core tasks in a **parallelized fashion** using Python's `multiprocessing` library. This means **porting this code over to any Map-Reduce framework shouldn't be too hard**. This also means **danny** can scale horizontally, which is great given the current shift towards cloud computing.

**danny** can either act as your **full ETL pipeline** for figuring out nearest neighbors **or** just a **useful tool to get quick neighbor information about certain users**. Further all the stages of **danny** can be completely used independently of the others, making the pipeline mode very flexible for others to drop in pieces they want to control... so if you're just interested in creating count dictionaries in a parallelized fashion, **danny** has you covered!

**Finally** I will be referring to a particular bipartite graph further down, here is what this graph is. You can look at the logs being passed in as the connections in a bipartite graph G(U, V, E), where:
* U = {set of all user vertices}
* V = {set of all entity vertices}
* E = {set of all edges between vertices in U and vertices in V}

Hopefully it is clear that this graph between users and entities is a bipartite one.

In general **danny** should be used when `O(|E|) << O(|U|*|V|)`

## How To Use

### Setup:
1. Clone this repo
2. Ensure you have pip installed (Note: Good practice would be to have a virtual-env for this project)
3. `sh setup.sh`

### How To Run:
1. Using the **danny** wrapper script (`python dannyw.py --help` will print out additional information) there are 6 different functionalities supported:

    1. **re\_index** : ensures your user and entity ids in your log file are consecutive ints starting from zero
    2. **dictionary** : builds the needed user_entity_dictionary and entity_user_dictionary from a properly formatted log file
    3. **matrix** : builds the needed user_entity_matrix from the user_entity_dictionary
    4. **nn** : by default will compute the nearest neighbors for all
    users in approximate mode. The number of nearest neighbors can either be a positive int below 1000, or -1 -- indicates no cap and to use the smart, but comprehensive mode of **danny**
    5. **build_index** : builds all three of the needed data structures for **danny** from a properly formatted log file. Essentially runs the "dictionary" and then "matrix" option.
    6. **batch** : computes nearest neighbors for each user from a properly 
    formatted log file. Essentially runs the "dictionary", "matrix" and finally "nn" options. Important to read how to configure the "nn" to your liking.

    For more information please read the doc-string at the top of dannyw.py file (proper documentation will be created soon)

2. `Import` **danny's** functionality into your own projects, the key functions to look at are:

    1. **supporting_functions.reindex_log_file**
    2. **supporting_functions.create_dictionaries**
    3. **supporting_functions.create_matrix**
    4. **dictionary_based_nn.prune_space_batch**
    5. **dictionary_based_nn.matrix_multiplication_batch**
    6. **dictionary_based_nn.get_nearest_neighbors_batch**

3. For ad-hoc work, running the **danny** wrapper in *build_index* mode and then using any of the three functions in **user_functions.py** can quickly give you nearest neighbor information on a handful of users. This avoids the full ETL pipeline functionality of danny, but still allows a researcher to gain valuable information on the nature of clusters and user behavior.

## Important File Descriptions
* `dannyw.py` - wrapper script that allows a user to interact with danny's core functionality via the command line.
* `supporting_functions.py` - all functionality needed by danny that isn't directly related to the nearest neighbor search. Functions to build danny's index and ensure the log file is of the needed format can be found here.
* `dictionary_based_nn.py` - all functionality pertinent to pruning the user space per user and computing dot products per user can be found here.
* `user_functions.py` - once danny's index is built you can start trying out quick experiments using these functions, instead of calculating similarities for all users (though danny does support partial batch operations).

## ETL Pipeline Description

**danny's** full ETL pipeline is as follows:

0. **danny** expects a log file in the format described above. However, danny also expects both user_ids and entity_ids to start from zero and be consecutive integers. If this is not the case, then the **first step** is to run **supporting_functions.reindex_log_file**. This will ensure your users and entities are indexed properly, as well as provide you a mapping between the old index and the new index.
    * Note: this is the only step not executed in **batch** mode of the `danny wrapper script`

1. **Construct the count/one-hot dictionaries:** From a properly formatted log file, **danny** will construct the dictionaries below. These dictionaries are used to prune the search space per user when looking for nearest neighbors. This dictionary creation is done in a parallel way.
    *  user-entity dictionary: key - user_id | value - dictionary
        * sub dictionary: key - entity_id | value - either total visits by user_id to entity_id or 1 for one hot encoding
    * entity-user dictionary: key - entity_id | value - dictionary
        * sub dictionary: key - user_id | value - either total visits by user_id to entity_id or 1 for one hot encoding

2. **Construct the user-entity count/one-hot matrix:** Using the *user-entity dictionary* **danny** constructs either a one_hot or count matrix, encoding the users' entity visitation patterns in the rows, and each entities' user visitation history in the columns. This uses sci-kit-learn's DictVectorizer function.

3. **Prune's User Space Per User:** Using the created dictionaries **danny** figures out per user which users share a common entity. **If user_i does not share an entity with user_j, then it makes little sense to compare their visitation patterns**. This pruning leverage Python's incredible dictionaries and per users runs in `O(avg_deg(u) * avg_deg(v))`. In **approximate mode**, **danny** spends a little more time pruning the space by **heuristically scoring** how likely each *entity-sharing-user's* visitation patterns will be to a given user's visitation pattern. After the scoring takes place (which adds no `big O` time), *entity-sharing-user's* are sorted and the first n are taken. This extra time spent pruning, allows **danny** to cap the amount of time spent per user in the dot product stage. Either way the result of this step is an array of the form below. **The pruning of the search space per user is written in a parallel way**
    * pruned_user_space_array: Each element in the array is the following tuple - `(user_id, list of relevant user_ids to check for that user)`

4. **Compute Dot Products:** Given a list of users to check per user, **danny parallelizes the task of computing dot products**. Each node has their own copy of the *user-entity-matrix* as well as a queue of `(user_id, list of relevant user_ids to check for that user)` tuples to work through. Slicing the matrix to only consider the relevant passed in users using `numpy`, **danny** computes only the needed dot products for each user. It returns these dot products in format below. From these dot products to select nearest neighbors is a trivial task. 
    * Approx Mode:
    * top-n-users-dictionary: key - user_id | value - dictionary
        * sub-dictionary: key - user_id | value: dot product
            * ^ length is always **n**
    + Exact Mode:
    + all-common-entity-users-dictionary: key | value - dictionary
        * sub-dictionary: key - user | value: dot product
            * ^ average length is **k**, where **k** = average number of users per user who share a common entity

*Note:* As **danny** will only compute **n** dot products when finding nearest neighbors in approximate mode, it is prudent to use a larger n than you will actually practically need for analysis / your pipeline. In this way you are covered if a request to expand the list of closest users per user comes in.

## Why Build danny:

I was playing around with Spotify's [annoy](https://github.com/spotify/annoy) package (which is great!), but the dataset I was had was causing *annoy* to take forever at the index building step. *Annoy* attempts to build up an LSH forest based on your user vectors which you can then use in a distributed manner to get nearest neighbors per user. In building an index, *annoy* is taking care of both the building up of a useful data structure for NN search, but also pruning the space of users to look at per user. This second part is what I suspect was causing *annoy* to take so long.

LSH forests are created by drawing hyperplanes through your vector space, which means in the calculations for how to break up this space you are looking at the full vector space and deciding how to slice it up. However, I knew that the dataset I was playing around represented a very sparse matrix, so I was frustrated by the need to look at all users at once. I was trying to think of a way to get around this, when I realized Python's dictionaries could help.

By pre-computing all entities associated with each user (user-entity-dictionary) and all users associated with each entity (entity-user-dictionary), you can figure out very quickly all users who share a entity with a given user. Look up the user_id in your user-entity-dictionary, grab all associated entities, then look up each entity_id from the associated entity list in the entity-user-dictionary to obtain all users that share an entity. There is hardly any point in comparing two users' visitation patterns if they've not shared a single entity, so using these dictionaries would allow me to prune the user space to only those users who have a shot of being close to a given user. So I decided dictionaries could act as the backbone of **danny's** index, as this way I would never be making calculations based on the whole user space.

**danny** also differs from *annoy* in the following ways:
* parallelizes 2/3rds of the tasks required to build its index
* parallelizes the task of pruning the user space
* the index is much simpler to create so quick testing takes less time to get up an running (user_functions.py)
* separates the processes of pruning the user space and building an index
    * if you only care about a subsection of the user space, **danny** would not have spent time partitioning parts of the user space you don't care about beforehand
* more transparency to what's happening at each stage, as well as the ability to use only parts of **danny's** pipeline
* does not use mmap memory (yet, that's one of my next things to work on) so will use up **more memory**

However, like *annoy* finding neighbors and computing the necessary dot products occurs in a parallelized way.

## Some Theory:
Using the bipartite graph framework described above we can look into `big O` complexity of **danny**

Here are some important facts to keep in mind about the bipartite graph:

    * Naive Nearest Neighbors runs in O(|U|^2 * |V|)
    * deg(u_i) = number of edges associated with user_i, lly for deg(v_i)
    * sum_over_all_users(deg(u_i)) = sum_over_all_entities(deg(v_i)) = |E|
    * |E| = sum_over_all_users(d(u)) = |U| * avg_d(u)
        * avg_d(u) = |E| / |U|, lly avg_d(v) = |E| / |V|
    * |U| <= |E|
    * |V| <= |E|
    * |E| <= |U| * |V|
    * When O(|E|) << O(|U|*|V|), deg(u_i) and deg(v_j) are low for most i, j

So as mentioned standard nearest neighbors occurs in `O(|U|*|U|*|V|)` time
* for each user you look at every other user and do |V| operations (the dot product)

So hows does danny do?

#### Building the index data structures:
* Building the dictionaries: `O(|E|)`
* Building the matrix: `O(|U|*|V|)`

#### In exact mode, we have the following Average Time complexity:
* let k = average number of users per user who share a common entity

* Pruning the space: `O(|U| * avg_d(u) * avg_d(v)) = O(|U|*(|E|/|U|)*(|E|/|V|)) = O(|E|^2/|V|)`

* Matrix multiplication: `O(|U|*|V|*k)`

So when `O(|E|) << O(|U|*|V|)`: (Note: |E| is maxed out at |U|*|V| in a bipartite graph)

    * O(|E|^2) <<<< O((|U|*|V|)^2)
    * therefore: O(|E|^2/|V|) <<<< O((|U|*|V|)^2 / |V|)
    * but (|U|*|V|)^2 / |V| == |U|^2*|V|
    * therefore: O(|E|^2/|V|) <<<< O(|U|^2*|V|)


    * also if O(k) << O(|U|)
        * which is highly likely given the degrees of each vertex (user and entity) is low 
    * then O(|U|*|V|*k) ~ O(|U|*|V|) << O(|U|^2*|V|)

    * both pruning and matrix multiplication steps takes less time than O(|U|^2*|V|)

#### In approximate mode, we have the following Average Time complexity:
* let k = average number of users per user who share a common entity

* Pruning the space: 
```
O(|U| * (avg_d(u) * avg_d(v) + klog(k)))
= O(|U|*((|E|/|U|)*(|E|/|V|) + klog(k)))
= O(|E|^2/|V| + |U|*klog(k))
```
           
* Matrix multiplication: `O(|U|*|V|)`

So when `O(|E|) << O(|U|*|V|)`: (Note: |E| is maxed out at |U|*|V| in a bipartite graph)

    * if O(k) << O(|U|)
        * which is highly likely given the degrees of each vertex (user and entity) is low
    * then O(|U|*klog(k)) ~ O(|U|)
    * then O(|E|^2/|V| + |U|*klog(k)) ~ O(|E|^2/|V| + |U|)
    * but O(|E|^2/|V|) <<<< O(|U|^2*|V|) (shown above)
    * and |U| <= |E| <= |E|^2/|E| <= |E|^2/|V|
    * therefore O(|E|^2/|V| + |U|) <<<< O(|U|^2*|V|)

    * O(|U|*|V|) << O(|U|^2*|V|)

    * both pruning and matrix multiplication steps takes less time than O(|U|^2*|V|)

So as you can see given the right conditions, `O(|E|) << O(|U|*|V|)`, **danny** preforms batter than regular Nearest Neighbors, with the additional benefit of running in parallel. For more on this check dictionary_based_nn.py.

## Work Still Left To Do:
1. Use mmap to reduce memory footprint
2. Add in examples with timing information
3. Create docs from doc-strings via sphinx
4. Try and reduce duplicated calculations arising from parallelization
5. Talk more about when to use exact mode and when to use approximate mode
6. Allow danny to be pip installable
7. Think about strategies to update the index and nearest neighbors as new users and entities enter the graph

## Copyright
Copyright (c) 2019 Rahul Khanna, released under the GPL v3 license.
