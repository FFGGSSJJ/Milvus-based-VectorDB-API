import time
import sys
import random

import logging
from pymilvus import connections, utility, DataType, \
    Collection, FieldSchema, CollectionSchema


# check usage
if (len(sys.argv) < 2):
    print("Usage: python3 clear_collection.py <host addr> <collection_name>")
    sys.exit()

# determine the host and collection name
host = sys.argv[1]  
collection_name = sys.argv[2]  
port = 19530

# connect
connections.add_connection(default={"host": host, "port": 19530})
connections.connect('default')

if (utility.has_collection(collection_name)):
    print("Find collection: going to release in memory and drop")
    collection = Collection(collection_name)
    collection.release()
    utility.drop_collection(collection_name)
    print("Done")
else:
    print("Collection not exist")

connections.disconnect('default')
