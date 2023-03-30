import time
import sys
import random

import logging
from pymilvus import connections, utility, DataType, \
    Collection, FieldSchema, CollectionSchema

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"

# check usage
if (len(sys.argv) < 4):
    print("Usage: python3 insert.py <host addr> <# of vectors> <dim# of vector>")
    sys.exit()

# determine the vector number and dimension by user
nb = int(sys.argv[2])    # 50000
dim = int(sys.argv[3])   # 128
auto_id = True
index_params1 = {"index_type": "IVF_PQ", "params": {"nlist": 128, "m": 2, "nbits": 8}, "metric_type": "L2"}
index_params2 = {"index_type": "IVF_SQ8", "params": {"nlist": 128}, "metric_type": "L2"}

if __name__ == '__main__':
    host = sys.argv[1]  # host address
    shards = 1          # shards number
    insert_times = 20   # insert times

    port = 19530
    connections.add_connection(default={"host": host, "port": 19530})
    connections.connect('default')
    log_name = "collection_prepare"

    logging.basicConfig(filename=f"/tmp/{log_name}.log",
                        level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)
    logging.info("=================start====================")
    collection_name = f"random_benchmark_collection"

    # check if collection exists
    if (utility.has_collection(collection_name)):
        logging.info('collection exists: release and drop the original collection')
        collection = Collection(collection_name)
        collection.release()
        utility.drop_collection(collection_name)

    # create collection with schemas
    id_field = FieldSchema(name="id", dtype=DataType.INT64,
                           description="auto primary id")
    age_field = FieldSchema(
        name="age", dtype=DataType.INT64, description="age")
    embedding_field = FieldSchema(
        name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    # testembedding_field = FieldSchema(
    #     name="test", dtype=DataType.FLOAT_VECTOR, dim=dim)
    schema = CollectionSchema(fields=[id_field, age_field, embedding_field],
                              auto_id=auto_id, primary_field=id_field.name,
                              description="my collection")
    collection = Collection(name=collection_name,
                            schema=schema, shards_num=shards)
    logging.info(f"create {collection_name} successfully")

    # check index info
    print("Index before first insert:")
    print([index.params for index in collection.indexes])

    # create index in the collection
    t0 = time.time()
    collection.create_index(field_name=embedding_field.name, index_params=index_params1)
    tt = round(time.time() - t0, 3)
    logging.info(f"Build index before 1st insert costs {tt}")

    # insert data into the collection
    for i in range(int(insert_times/2)):
        # prepare data
        ages = [random.randint(1, 100) for _ in range(nb)]
        embeddings = [[random.random() for _ in range(dim)] for _ in range(nb)]
        # embeddings = [[] for _ in range(nb)]
        data = [ages, embeddings]
        t0 = time.time()
        collection.insert(data)
        tt = round(time.time() - t0, 3)
        logging.info(f"Insert data{i}:[[{nb}*1], [{nb}*{dim}]] costs {tt}s")

    collection.flush()
    logging.info(f"collection entities: {collection.num_entities}")

    # create index in the collection
    t0 = time.time()
    collection.create_index(field_name=embedding_field.name, index_params=index_params1)
    tt = round(time.time() - t0, 3)
    logging.info(f"Build index after 1st insert costs {tt}")

    # check index info
    print("Index after first insert:")
    print([index.params for index in collection.indexes])

    # insert again
    for i in range(int(insert_times/2)):
        # prepare data
        ages = [random.randint(1, 100) for _ in range(nb)]
        embeddings = [[random.random() for _ in range(dim)] for _ in range(nb)]
        data = [ages, embeddings]
        t0 = time.time()
        collection.insert(data)
        tt = round(time.time() - t0, 3)
        logging.info(f"Insert data{i}:[[{nb}*1], [{nb}*{dim}]] costs {tt}s")

    # check index info
    print("Index after second insert:")
    print([index.params for index in collection.indexes])

    # create index in the collection
    t0 = time.time()
    collection.create_index(field_name=embedding_field.name, index_params=index_params1)
    tt = round(time.time() - t0, 3)
    logging.info(f"Build index after 2nd insert costs {tt}")

    # drop
    if (utility.has_collection(collection_name)):
        print("Find collection: going to release in memory and drop")
        collection = Collection(collection_name)
        collection.release()
        utility.drop_collection(collection_name)
        print("Done")
    else:
        print("Collection not exist")

    connections.disconnect('default')

    # load the collection into memory
    # collection.load()
    # logging.info("collection prepare completed")

    # disconnect ?
