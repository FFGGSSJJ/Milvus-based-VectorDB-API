package milvusproxy

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/bwmarrin/snowflake"
	"github.com/gammazero/workerpool"
	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/nsqio/go-nsq"
	log "github.com/sirupsen/logrus"
	"github.com/spf13/viper"
	"google.golang.org/protobuf/proto"
	pb "vision.foovideo.cn/lookonce/internal/pkg/foovideo"
	"vision.foovideo.cn/lookonce/internal/pkg/vearchapi"
)

type MilvusServer struct {
	milvusClient   client.Client
	bnnWithAttrLen int
}

type MilvusResultItem struct {
	FeatureID    int64
	ObjectID     int64
	Score        float32
	FeatureGroup string
	GroupID      string
	ObjectType   uint8
	Timestamp    uint32
}

type MilvusQueryItem struct {
	FeatureID  int64
	FeatureVec []float32
}

type MilvusSearchResult struct {
	FeatureIndex int // 多特征检索时，属于哪一个特征的结果
	TotalHits    int32
	// FeatureGroup string
	FeatureItems []MilvusResultItem
}

type MilvusQueryResult struct {
	FeatureIndex int // 多特征检索时，属于哪一个特征的结果
	FeatureItems []MilvusQueryItem
	DistMap      map[int64]float64
}

type MilvusSearchReq struct {
	CollName       string
	PartitionNames []string
	OutputFields   []string
	Feature        []float32
	VecName        string
	Expr           string
	// MetricType     entity.MetricType
	Nprobe int32
	TopK   int32
}

func (s *MilvusServer) milvusStartNsqConsumer(reidAttrAllDim, reidAttrDim int) {

	// Clear all collections in Milvus at the very beginning
	// TODO: remove this part, it is for debug only
	// s.milvusListAllCollections()
	// s.milvusGetIndexState()
	s.milvusDescribeCollections()
	// s.milvusIndexTest()
	// s.milvusLoadTest()
	// s.milvusDropAllCollection()
	// return

	// Create Collections in Milvus

	//nsqAddr := viper.GetString("nsq.nsqaddr")
	log.Println("Milvus Start NsqConsumer..")
	config := nsq.NewConfig()
	config.MaxInFlight = 1000
	config.DefaultRequeueDelay = 15
	config.MaxAttempts = 10

	q, _ := nsq.NewConsumer("MediaData", "MilvusFeatures", config)
	// producer, _ := nsq.NewProducer(nsqAddr, config)
	// failedTopic := "MilvusSaveFailedData"

	// add handler for handling message
	log.Println("Add Concurrent Handlers")
	q.AddConcurrentHandlers(nsq.HandlerFunc(func(message *nsq.Message) error {
		// get msg from queue and convert to FrameInfo
		frameInfo := pb.ParserFrameInfo{}
		err := proto.Unmarshal(message.Body, &frameInfo)

		if err != nil {
			log.Error("数据解析失败: ", err, message.Body)
		} else {
			go func() {
				// loop the frameinfo
				for _, finfo := range frameInfo.RealtimePaths {
					// parse the pathid
					pathID, err := strconv.ParseInt(finfo.PathId, 10, 64)
					if pathID == 0 || err != nil {
						log.Errorf("id解析出错: %v, %v,%v", frameInfo.TaskId, finfo.FeatureId, err)
						continue
					}

					// extract Face related data from frameinfo and store
					if finfo.FaceParserMetaData != nil && finfo.FaceParserMetaData.FeatureId > 0 && len(finfo.FaceParserMetaData.Feature) > 0 {
						err := s.milvusStoreFeatureToCollection(pb.AlgorithmVersion_VERSION_FACE, finfo.FrameTimestamp, uint64(finfo.FaceParserMetaData.FeatureId), pathID, frameInfo.TaskId,
							finfo.FaceParserMetaData.Feature, pb.ObjectType_OBJECTTYPE_PED, finfo.FeatureGroupId)
						if err != nil {
							log.Errorf("人脸数据存储失败：", frameInfo.TaskId, finfo.FeatureId, err)
							return
						}
						log.Println("人脸数据存储成功：", frameInfo.TaskId, finfo.FeatureId, pathID)
					}

					// extract ReidHeadAttrFeature from frameinfo and store
					if len(finfo.ReidHeadAttrFeature) > 0 {
						err := s.milvusStoreFeatureToCollection(pb.AlgorithmVersion_VERSION_REID_HEAD_ATTR, finfo.FrameTimestamp, uint64(finfo.FeatureId), pathID, frameInfo.TaskId,
							finfo.ReidHeadAttrFeature, finfo.Type, finfo.FeatureGroupId)
						if err != nil {
							log.Errorf("形体数据存储失败：", frameInfo.TaskId, finfo.FeatureId, err)
							return
						}
						log.Println("形体数据存储成功：", frameInfo.TaskId, finfo.FeatureId, pathID)
					}
				}

			}()
		}
		return nil
	}), 16)

	// connect to nsq
	lookups := strings.Split(viper.GetString("nsq.nsqlookupaddr"), "=") // deis-nsqlookupd-0.deis-nsqlookupd.mq:4161,deis-nsqlookupd-1.deis-nsqlookupd.mq:4160,deis-nsqlookupd-2.deis-nsqlookupd.mq
	log.Info("connect to NSQ lookups: ", lookups)
	err := q.ConnectToNSQLookupds(lookups)
	if err != nil {
		log.Panic("连接到nsq失败", lookups)
	}
}

/****************************** Milvus Data Store *************************************/

/**
 * @description: Store Data into Milvus
 * @param {pb.AlgorithmVersion} alg
 * @param {*} timestamp
 * @param {uint64} featureID
 * @param {int64} pathID
 * @param {string} taskid
 * @param {[]byte} feature
 * @param {pb.ObjectType} objType
 * @param {string} groupID
 * @return {*}
 */
// TODO: extract schema create part as a single function that could accept self-defined fields
func (s *MilvusServer) milvusStoreFeatureToCollection(alg pb.AlgorithmVersion, timestamp, featureID uint64, pathID int64, taskid string, feature []byte,
	objType pb.ObjectType, groupID string) error {
	ctx := context.Background()
	// check algorithm type
	if alg != pb.AlgorithmVersion_VERSION_FACE && alg != pb.AlgorithmVersion_VERSION_REID_HEAD_ATTR {
		log.Errorf("算法类型错误: %v, %v", alg, featureID)
	}

	// check feature cutdown:  形体特征与融合特征，添加截断的操作
	if (alg == pb.AlgorithmVersion_VERSION_BNN_PRO_ATTR_SCORE || alg == pb.AlgorithmVersion_VERSION_REID_HEAD_ATTR) && len(feature) > s.bnnWithAttrLen {
		feature = feature[0:s.bnnWithAttrLen]
	}

	// generate collection name according to algorithm
	var collection_name string
	switch alg {
	case pb.AlgorithmVersion_VERSION_FACE:
		collection_name = "VERSION_FACE"
	case pb.AlgorithmVersion_VERSION_REID_HEAD_ATTR:
		collection_name = "VERSION_REID_HEAD_ATTR"
	default:
		collection_name = "VERSION_UNKNOWN"
	}

	// check online/offline using group_id
	_, timetype, _, _ := vearchapi.ParseGroupID(groupID)
	// log.Println("Partition Type: ", timetype)

	// generate partition name according to feature_id
	var partition_name string
	if timetype == "realtime" {
		feature_timestamp := snowflake.ParseInt64(int64(featureID))
		partition_time := time.Unix(feature_timestamp.Time()/1000, 0).Local()
		partition_name = fmt.Sprintf("p_%d%02d%02d", partition_time.Year(), partition_time.Month(), partition_time.Day())
		// log.Println("Realtime Partition Name: ", partition_name)
	} else {
		partition_name = "offline_partition"
	}

	// check if collection exist
	collection_exists, _ := s.milvusClient.HasCollection(ctx, collection_name)

	// if collection already exists
	if collection_exists {
		// insert data
		_ = s.milvusInsertVectorByCol(ctx, collection_name, partition_name, int64(featureID), int64(pathID), int64(timestamp), taskid, feature)

		// flush collection
		err := s.milvusClient.Flush(ctx, collection_name, false)
		if err != nil {
			log.Errorf("milvus flush failed: %v", err)
			return err
		}
	}

	// if collection does not exists
	if !collection_exists {
		var vec_dim string
		if collection_name == "VERSION_FACE" {
			vec_dim = "256"
		} else {
			vec_dim = "272"
		}
		// create schema for collection
		schema := &entity.Schema{
			CollectionName: collection_name,
			Description:    "Collection for algorithm " + collection_name,
			AutoID:         false,
			Fields: []*entity.Field{
				{
					Name:        "Feature_id",
					PrimaryKey:  true,
					AutoID:      false,
					DataType:    entity.FieldTypeInt64,
					Description: "primary key field",
				},
				{
					Name:       "Task_id",
					PrimaryKey: false,
					AutoID:     false,
					DataType:   entity.FieldTypeVarChar,
					TypeParams: map[string]string{
						"max_length": "1024",
					},
					Description: "",
				},
				{
					Name:        "Timeinfo",
					PrimaryKey:  false,
					AutoID:      false,
					DataType:    entity.FieldTypeInt64,
					Description: "",
				},
				{
					Name:        "Object_id",
					PrimaryKey:  false,
					AutoID:      false,
					DataType:    entity.FieldTypeInt64,
					Description: "",
				},
				{
					Name:       "Features",
					PrimaryKey: false,
					AutoID:     false,
					DataType:   entity.FieldTypeFloatVector,
					TypeParams: map[string]string{
						"dim": vec_dim,
					},
					Description: "",
				},
			},
		}

		// create collection with schema
		success, err := s.milvusCreateCollection(ctx, collection_name, schema, 2)
		if !success {
			log.Errorf("milvus create collection failed: %v", err)
			// return err
		}

		// insert data
		_ = s.milvusInsertVectorByCol(ctx, collection_name, partition_name, int64(featureID), int64(timestamp), int64(pathID), taskid, feature)

		// flush collection
		// err = s.milvusClient.Flush(ctx, collection_name, false)
		// if (err != nil) {
		// 	log.Errorf("milvus flush failed: %v", err)
		// 	return err
		// }

		// create index
		err = s.milvusCreateVectorIndex(ctx, collection_name)
		if err != nil {
			log.Errorf("milvus index create failed: %v", err)
			return err
		}
	}

	return nil
}

/**
 * @description: create collection by name
 * @param {string} collName: name of the collection to create
 * @param {*entity.Schema} schema: schema to create the collection
 * @param {int} shardnum: number of shard in collection
 * @return {*}
 */
func (s *MilvusServer) milvusCreateCollection(ctx context.Context, collName string, schema *entity.Schema, shardnum int) (bool, error) {
	err := s.milvusClient.CreateCollection(ctx, schema, int32(shardnum))
	if err != nil {
		log.Errorf("milvus collection create failed: %v", err)
		return false, err
	}
	return true, nil
}

/**
 * @description: create partition by name in specified collection
 * @param {string} collection_name
 * @param {string} partition_name
 * @return {*}
 */
func (s *MilvusServer) milvusCreatePartition(ctx context.Context, collection_name string, partition_name string) error {
	// check collection
	collection_exists, err := s.milvusClient.HasCollection(ctx, collection_name)
	if err != nil || !collection_exists {
		log.Errorf("milvus partition create failed: %v", err)
		return err
	}

	// create partition
	err = s.milvusClient.CreatePartition(ctx, collection_name, partition_name)
	if err != nil {
		log.Errorf("milvus partition create failed: %v", err)
		return err
	}
	log.Debug("Partition Created:", partition_name)
	return nil
}

/**
 * @description: create IVF_PQ index in specified collection
 * @param {string} collection_name
 * @return {*}
 */
// TODO: support multiple index creation and index parameters
func (s *MilvusServer) milvusCreateVectorIndex(ctx context.Context, collection_name string) error {
	vecidx, err := entity.NewIndexIvfPQ(entity.L2, 2048, 16, 8)
	if err != nil {
		log.Errorf("milvus get index failed: %v", err)
		return err
	}

	err = s.milvusClient.CreateIndex(ctx, collection_name, "Features", vecidx, false)
	if err != nil {
		log.Errorf("milvus create index failed: %v", err)
		return err
	}

	return nil
}

// insert a single vector into Milvus database
func (s *MilvusServer) milvusInsertVectorByCol(ctx context.Context, collection_name string, partition_name string, featureID int64, timestamp int64, pathID int64, taskID string, feature []byte) error {
	// check partition
	has_partition, err := s.milvusClient.HasPartition(ctx, collection_name, partition_name)
	if err == nil && !has_partition {
		err = s.milvusCreatePartition(ctx, collection_name, partition_name)
		if err != nil {
			log.Errorf("milvus insert single vector failed due to: %v", err)
			// return err
		}
	}

	// create columns
	featureids := []int64{featureID}
	featureidCol := entity.NewColumnInt64("Feature_id", featureids)

	objectids := []int64{pathID}
	objectidCol := entity.NewColumnInt64("Object_id", objectids)

	timestamps := []int64{timestamp}
	timestampCol := entity.NewColumnInt64("Timeinfo", timestamps)

	taskids := []string{taskID}
	taskidCol := entity.NewColumnVarChar("Task_id", taskids)

	features := make([][]float32, 0, 1)
	float_features := ByteToFloat32Array(feature, len(feature)/4)
	features = append(features, float_features)
	featureCol := entity.NewColumnFloatVector("Features", len(feature)/4, features)

	// insert data into collection
	_, err = s.milvusClient.Insert(ctx, collection_name, partition_name, featureidCol, objectidCol, timestampCol, taskidCol, featureCol)
	if err != nil {
		log.Errorf("milvus insert failed: %v", err)
		return err
	}
	return nil
}

// TODO:perform ranking for returned results
func (s *MilvusServer) milvusRanking(ctx context.Context, results *[]MilvusSearchResult, req MilvusSearchReq) (*[]MilvusSearchResult, error) {
	// perform query to get vectors
	qres, err := s.milvusRetriveVectors(ctx, results, req, true)
	if err != nil {
		log.Error("milvusRanking failed: ", err)
		return nil, err
	}

	if len(qres) == 0 {
		log.Println("milvusRanking abandoned")
		return results, nil
	}

	return nil, nil
}

func (s *MilvusServer) milvusRetriveVectors(ctx context.Context, results *[]MilvusSearchResult, req MilvusSearchReq, calcRankingDis bool) ([]MilvusQueryResult, error) {
	rets := []MilvusQueryResult{}
	pool := workerpool.New(16)
	mtx := &sync.Mutex{}

	var algo pb.AlgorithmVersion
	switch req.CollName {
	case "VERSION_FACE":
		algo = pb.AlgorithmVersion_VERSION_FACE
	case "VERSION_REID_HEAD_ATTR":
		algo = pb.AlgorithmVersion_VERSION_REID_HEAD_ATTR
	}

	for id, res := range *results {
		qitems := []MilvusQueryItem{}

		// pack req
		qreqs := balanceQueryReq(20, &res, req)
		// retrive QueryItems
		for _, qreq := range qreqs {
			pool.Submit(func() {
				_, qitem, err := s.milvusQuery(ctx, qreq)

				if err != nil {
					log.Error("milvusRetriveVectors failed: ", err)
					return
				}
				mtx.Lock()
				qitems = append(qitems, qitem...)
				mtx.Unlock()
			})
		}
		pool.StopWait()

		// pack to QueryResult
		ret := MilvusQueryResult{
			FeatureIndex: id,
			FeatureItems: qitems,
			DistMap:      make(map[int64]float64),
		}

		if calcRankingDis {
			for _, qitem := range qitems {
				dist := calcRankingL2Dis(req.Feature, qitem.FeatureVec, len(req.Feature))
				ret.DistMap[qitem.FeatureID] = dist
			}
			for _, sitem := range (&res).FeatureItems {
				log.Println("Score before rank::", sitem.Score)
				(&sitem).Score = calcScore(float32(ret.DistMap[sitem.FeatureID]), algo)
				log.Println("Score after rank::", sitem.Score)
			}
		}

		rets = append(rets, ret)
	}

	return rets, nil
}

// query and return query result
func (s *MilvusServer) milvusQuery(ctx context.Context, req MilvusSearchReq) ([]entity.Column, []MilvusQueryItem, error) {
	// perform query
	cols, err := s.milvusClient.Query(
		ctx,
		req.CollName,
		req.PartitionNames,
		req.Expr,
		req.OutputFields,
	)
	if err != nil {
		log.Error("Milvus Query faield: ", err)
		return nil, nil, err
	}

	// parse returned result and pack into MilvusQueryResult
	var featureidCol *entity.ColumnInt64
	var featureCol *entity.ColumnFloatVector
	for _, col := range cols {
		switch col.Name() {
		case "Feature_id":
			c, ok := col.(*entity.ColumnInt64)
			if ok {
				featureidCol = c
			}
		case "Features":
			c, ok := col.(*entity.ColumnFloatVector)
			if ok {
				featureCol = c
			}
		}
	}

	result := []MilvusQueryItem{}
	log.Println("Query DEBUG::")
	log.Println("\tfeatureidCol len::", featureidCol.Len())

	for i := 0; i < featureidCol.Len(); i++ {
		fid, _ := featureidCol.ValueByIdx(i)
		res := MilvusQueryItem{
			FeatureID:  fid,
			FeatureVec: featureCol.Data()[i],
		}
		result = append(result, res)
	}

	return cols, result, nil
}

/****************************** Milvus API Functions *************************************/

/*
* @description: initialize milvus connection

* @return {*MilvusServer}
 */
func Init() *MilvusServer {
	var s MilvusServer
	log.Println("Milvus Try Connect")
	// connect to the milvus server
	milvusClient, err := client.NewGrpcClient(context.Background(), "10.109.219.126:19530")
	if err != nil {
		log.Fatal("Milvus connect failed: %s", err.Error())
		return nil
	}
	log.Println("Milvus Connected")
	s.milvusClient = milvusClient
	reidAttrAllDim := 272
	s.bnnWithAttrLen = 4 * reidAttrAllDim
	return &s
}

/**
 * @description: check collection by name
 * @param {context.Context} ctx
 * @param {string} collection_name
 * @return {*}
 */
func (s *MilvusServer) MilvusCheckCollection(ctx context.Context, collection_name string) (bool, error) {
	log.Println("Check Collection: ", collection_name)
	return s.milvusClient.HasCollection(ctx, collection_name)
}

/*
 * @description: check partition by name
 * @param {*} ctx
 * @param {*} collection_name
 * @param {*} partition_name
 * @return {*}
 */
func (s *MilvusServer) MilvusCheckPartition(ctx context.Context, collection_name, partition_name string) (bool, error) {
	return s.milvusClient.HasPartition(ctx, collection_name, partition_name)
}

/*
 * @description: perform milvus search
 * @param {context.Context} ctx
 * @param {MilvusSearchReq} req
 * @param {[]string} outputFields
 * @return {*}
 */
// TODO modify hard coded part :: metric type
func (s *MilvusServer) MilvusSearchFeatures(ctx context.Context, req MilvusSearchReq) ([]MilvusSearchResult, error) {
	// ahead of searching, load target partitions into memory
	// err := s.milvusClient.LoadPartitions(ctx, req.CollName, req.PartitionNames, false)
	// if err != nil {
	// 	log.Error("Milvus load partitions failed: ", err)
	// 	return nil, err
	// }

	// prepare for search
	results := []MilvusSearchResult{}
	sp, _ := entity.NewIndexIvfPQSearchParam(int(req.Nprobe))
	vec := []entity.Vector{
		entity.FloatVector(req.Feature),
	}

	// perform search
	Rets, err := s.milvusClient.Search(
		ctx,
		req.CollName,
		req.PartitionNames,
		req.Expr,
		req.OutputFields,
		vec,
		req.VecName,
		entity.L2,
		int(req.TopK),
		sp,
	)

	var algo pb.AlgorithmVersion
	switch req.CollName {
	case "VERSION_FACE":
		algo = pb.AlgorithmVersion_VERSION_FACE
	case "VERSION_REID_HEAD_ATTR":
		algo = pb.AlgorithmVersion_VERSION_REID_HEAD_ATTR
	}

	// handle the search results
	if err != nil {
		log.Error("Milvus search failed: ", err)
		return nil, err
	} else {
		results = milvusPackResults(Rets, req.OutputFields, algo)
	}

	// test
	qres, err := s.milvusRetriveVectors(ctx, &results, req, true)
	for _, q := range qres {
		log.Println("FeatureID: ", q.FeatureIndex)
		log.Println("FeatureVCnum: ", len(q.FeatureItems))
	}

	return results, nil
}

/****************************** Milvus Help Functions *************************************/

func ByteToFloat32(bytes []byte) float32 {
	bits := binary.LittleEndian.Uint32(bytes)
	return math.Float32frombits(bits)
}

func ByteToFloat32Array(bf []byte, featureSize int) []float32 {
	feature := make([]float32, featureSize)
	for i := 0; i < featureSize; i++ {
		off := i * 4
		if len(bf) <= off || len(bf) <= off+4 {
			continue
		}
		feature[i] = ByteToFloat32(bf[off : off+4])
	}
	return feature
}

/*
* 1. how should TotalHit field be set
* 2. modify hard coded part::outputFields
* 3. taskid unused
 */
// TODO: implement milvusPackResults::support VERSION_FACE
func milvusPackResults(Rets []client.SearchResult, outputFields []string, algo pb.AlgorithmVersion) []MilvusSearchResult {
	results := []MilvusSearchResult{}

	for id, res := range Rets {
		var featureidCol *entity.ColumnInt64
		var objectidCol *entity.ColumnInt64
		var timeCol *entity.ColumnInt64
		var taskidCol *entity.ColumnVarChar
		// log.Println("Parse Fields::")
		for _, field := range res.Fields {

			// log.Println("\tFields::", field.Name())
			switch field.Name() {
			case "Feature_id":
				c, ok := field.(*entity.ColumnInt64)
				if ok {
					featureidCol = c
				}
			case "Object_id":
				c, ok := field.(*entity.ColumnInt64)
				if ok {
					objectidCol = c
				}
			case "Timeinfo":
				c, ok := field.(*entity.ColumnInt64)
				if ok {
					timeCol = c
				}
			case "Task_id":
				c, ok := field.(*entity.ColumnVarChar)
				if ok {
					taskidCol = c
				}
			default:
				continue
			}
		}

		// construct search result
		result := MilvusSearchResult{}

		// DEBUG
		log.Println("DEBUG::")
		log.Println("\tSearch Result Len: ", featureidCol.Len())
		for i := 0; i < res.ResultCount; i++ {
			featureid, err := featureidCol.ValueByIdx(i)
			objectid, err := objectidCol.ValueByIdx(i)
			timestamp, err := timeCol.ValueByIdx(i)
			_, _ = taskidCol.ValueByIdx(i)
			if err != nil {
				log.Error("milvusPackResults failed: ", err.Error())
			}

			log.Println("\t\tResult::", i, " ", featureid, " ", res.Scores[i])
			// construct return item
			item := MilvusResultItem{
				FeatureID:    featureid,
				ObjectID:     objectid,
				Score:        calcScore(res.Scores[i], algo),
				FeatureGroup: "",
				GroupID:      "",
				ObjectType:   0,
				Timestamp:    uint32(timestamp),
			}
			result.FeatureItems = append(result.FeatureItems, item)
		}
		result.FeatureIndex = id
		result.TotalHits = 0 // ???
		results = append(results, result)
	}
	return results
}

func balanceQueryReq(burst int32, result *MilvusSearchResult, req MilvusSearchReq) []MilvusSearchReq {
	reqs := []MilvusSearchReq{}
	fnum := len(result.FeatureItems)
	bnum := fnum / int(burst)
	offset := fnum % int(burst)

	// pack burst req
	for i := 0; i < bnum; i++ {
		r := MilvusSearchReq{
			CollName:       req.CollName,
			PartitionNames: req.PartitionNames,
			Expr:           "",
			OutputFields:   []string{"Feature_id", "Features"},
		}
		for j := i * int(burst); j < (i+1)*int(burst); j++ {
			r.Expr = r.Expr + "Feature_id == " + strconv.Itoa(int(result.FeatureItems[j].FeatureID))
			if j < (i+1)*int(burst)-1 {
				r.Expr = r.Expr + " || "
			}
		}
		reqs = append(reqs, r)
	}

	// pack offset req
	if offset > 0 {
		r := MilvusSearchReq{
			CollName:       req.CollName,
			PartitionNames: req.PartitionNames,
			Expr:           "",
			OutputFields:   []string{"Feature_id", "Features"},
		}
		for i := bnum * int(burst); i < bnum*int(burst)+offset; i++ {
			r.Expr = r.Expr + "Feature_id == " + strconv.Itoa(int(result.FeatureItems[i].FeatureID))
			if i < bnum*int(burst)+offset-1 {
				r.Expr = r.Expr + " || "
			}
		}
		reqs = append(reqs, r)
	}

	return reqs
}

func calcRankingL2Dis(src, dst []float32, dim int) float64 {
	var sqsum float64
	sqsum = 0
	for i := 0; i < dim; i++ {
		sqsum = sqsum + float64((src[i]-dst[i])*(src[i]-dst[i]))
	}
	return math.Sqrt(sqsum)
}

func calcScore(dis float32, algo pb.AlgorithmVersion) float32 {
	var res float32
	switch algo {
	case pb.AlgorithmVersion_VERSION_FACE:
		res = float32(dis)
	case pb.AlgorithmVersion_VERSION_REID_HEAD_ATTR:
		newdisNormalization := 2 + 0.2*6.3442
		newDis := float64(dis) / newdisNormalization
		res = float32(math.Exp(float64(-1.0 * math.Acos(-1.0) * math.Pow(newDis, 2))))
	}

	return res
}

func Run() {
	s := Init()
	// reidAttrAllDim, reidAttrDim, _ := vearchapi.VearchAPIInstance().FeatureDim(pb.AlgorithmVersion_VERSION_REID_HEAD_ATTR)
	var reidAttrAllDim, reidAttrDim int
	reidAttrAllDim = 272
	reidAttrDim = 272
	s.bnnWithAttrLen = 4 * reidAttrAllDim
	s.milvusStartNsqConsumer(reidAttrAllDim, reidAttrDim)
}
