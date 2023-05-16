package milvusproxy
import (
	"context"
	log "github.com/sirupsen/logrus"
)




/****************************** Milvus Test Functions *************************************/
/*
 * @description: list all collections in Milvus
 * @return {*}
 */
// ! [Debug Only]
func (s *MilvusServer) milvusListAllCollections() {
	// list all collection in Milvus
	colList, err := s.milvusClient.ListCollections(context.Background())
	if err != nil {
    	log.Error("fail to get collection list: ", err.Error())
		return
	}
	
	// loop to list all collection
	for i := 0; i < len(colList); i++ {
		colName := colList[i].Name
		log.Println("Collection : ", i," ",colName)
	}
}

/*
 * @description: drop all collections in Milvus
 * @return {*}
 */
// ! [Debug Only]
func (s *MilvusServer) milvusDropAllCollection() {
	// list all collection in Milvus
	colList, err := s.milvusClient.ListCollections(context.Background())
	if err != nil {
    	log.Error("fail to get collection list: ", err.Error())
		return
	}
	
	// loop to drop all collection
	for i := 0; i < len(colList); i++ {
		colName := colList[i].Name
		log.Println("Collection drop: ", colName)
		err = s.milvusClient.DropCollection(context.Background(), colName)
		if err != nil {
    		log.Error("fail to drop collection: ", err.Error())
		}
	}
}


/*
 * @description: get index building state of all collections
 * @return {*}
 */
 // ! [Debug Only]
func (s *MilvusServer) milvusGetIndexState() {
	// get all collection in Milvus
	colList, err := s.milvusClient.ListCollections(context.Background())
	if err != nil {
    	log.Error("fail to get collection list: ", err.Error())
		return
	}

	// loop to get collection index state
	for i := 0; i < len(colList); i++ {
		colName := colList[i].Name
		
		indexProgress, err := s.milvusClient.GetIndexState(
			context.Background(),
			colName,
			"Features",
		)

		if err != nil {
			log.Error("fail to get index state of ", colName, ":", err.Error())
			return
		} else {
			log.Println("Index state of ", colName, ": %v\n", indexProgress)
		}
	}
}

/*
 * @description: describe all collections information: [name, channels, shard num, index state]
 * @return {*}
 */
 // ! [Debug Only]
func (s *MilvusServer) milvusDescribeCollections() {
	// get all collection in Milvus
	colList, err := s.milvusClient.ListCollections(context.Background())
	if err != nil {
    	log.Error("fail to get collection list: ", err.Error())
		return
	}

	// describe collections
	for _, coll := range colList {
		indexProgress, err := s.milvusClient.GetIndexState(
			context.Background(),
			coll.Name,
			"Features",
		)
		if err != nil {
			log.Error("fail to get index state of ", coll.Name, ":", err.Error())
			return
		}

		// describe
		log.Println("-Collection Name: ", coll.Name)
		// log.Panicln("\tLoaded: ", int(coll.Loaded))
		if coll.PhysicalChannels != nil {
			for i , name := range coll.PhysicalChannels {
				log.Println("\tPChannel: ", i, "", name)
			}
		} else {
			log.Println("\tPChannel: NULL")
		}

		if coll.VirtualChannels != nil {
			for i , name := range coll.VirtualChannels {
				log.Println("\tVChannel: ", i, "", name)
			}
		} else {
			log.Println("\tVChannel: NULL")
		}

		log.Println("\tShardNum: ", coll.ShardNum)
		log.Println("\tIndex State: ", indexProgress)
	}
}


func (s *MilvusServer) milvusIndexTest() {
	// get all collection in Milvus
	log.Println("Milvus Load Test --")
	colList, err := s.milvusClient.ListCollections(context.Background())
	if err != nil {
    	log.Error("fail to get collection list: ", err.Error())
		return
	}

	for _, coll := range colList {
		err := s.milvusCreateVectorIndex(context.Background(), coll.Name)
		if err != nil {
			log.Error("milvusIndexTest failed: ", coll.Name, " ", err)
			continue
		}
	}
}


func (s *MilvusServer) milvusLoadTest() {
	// get all collection in Milvus
	log.Println("Milvus Load Test --")
	colList, err := s.milvusClient.ListCollections(context.Background())
	if err != nil {
    	log.Error("fail to get collection list: ", err.Error())
		return
	}

	for _, coll := range colList {
		// err := s.milvusClient.LoadCollection(context.Background(), coll.Name, true)
		err := s.milvusClient.LoadPartitions(context.Background(), coll.Name, []string{"p_20230410"}, false)
		if err != nil {
			log.Error("milvusLoadTest failed: ", coll.Name, " err::", err)
			continue
		} else {
			log.Println("Collection ", coll.Name, " Loaded")
			// err := s.milvusClient.ReleaseCollection(context.Background(), coll.Name)
			err := s.milvusClient.ReleasePartitions(context.Background(), coll.Name, []string{"p_20230410"})
			if err != nil {
				log.Error("milvusLoadTest failed: ", coll.Name, " err::", err)
				continue
			}
		}
	}
}