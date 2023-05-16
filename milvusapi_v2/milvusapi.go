package milvusapi

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
	"regexp"
	"sync"
	"time"

	// "github.com/gammazero/workerpool"
	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	log "github.com/sirupsen/logrus"
	"github.com/spf13/viper"
	// featurepb "polaris.ihouqi.cn/pkg/genproto/polaris/inner/feature/v1"
)

var (
	apiOnce                    sync.Once
	api                        *MilvusServer
	CollectionNameFormatRegexp = regexp.MustCompile(COLLECTION_NAME_FORMAT)
)

const (
	// collection name format in milvus
	COLLECTION_NAME_FORMAT = "^(ped_feat|ped_attr|face_feat|bike_feat|bike_attr|vehicle_feat)_([0-9]+)(_([0-9]+))?$"
	NProbe                 = 2048
	IndexEntityThreshold   = 200 // not used
)

type MilvusServer struct {
	milvusClient       client.Client
	collectionLoadMap  map[string]bool
	collectionIndexMap map[string]bool

	// TODO: to be optimized.
	// These fields are used to avoid concurrent creation of collection/partition.
	// I haven't figured out a better approach
	initCollection map[string]bool
	initPartition  map[string]bool
	collMtx        *sync.Mutex
	partMtx        *sync.Mutex
}

type MilvusSearchItem struct {
	FeatureID int64
	JobID     int64
	TrackID   int64
	Timestamp int64
	Score     float32
}

type MilvusSearchResult struct {
	FeatureIndex int // 多特征检索时，属于哪一个特征的结果
	TotalHits    int32
	// FeatureGroup string
	FeatureItems []MilvusSearchItem
}

type MilvusQueryItem struct {
	CollName   string
	FeatureID  int64
	FeatureVec []float32
}

type MilvusQueryResult struct {
	FeatureIndex int // 多特征检索时，属于哪一个特征的结果
	FeatureItems []MilvusQueryItem
	DistMap      map[int64]float64
}

// Both Search and Query use this as req
type MilvusSearchReq struct {
	CollName       string
	PartitionNames []string
	OutputFields   []string
	Feature        [][]float32
	VecName        string
	Expr           string
	MetricType     entity.MetricType
	Nprobe         int32
	TopK           int32
	NoRanking      bool
	ScoreThreshold float32
}

// NewMilvusAPI create a instance of MilvusAPI
// ! call this only once unless necessary
func NewMilvusAPI() *MilvusServer {
	api := &MilvusServer{}
	return api
}

// MilvusAPIInstance singleton func for MilvusAPI
func MilvusAPIInstance() *MilvusServer {
	apiOnce.Do(func() {
		api = NewMilvusAPI()
	})
	return api
}

/****************************** Milvus Search Result func *************************************/

/**
 * @description: generate map[job] count
 * @return {*}
 */
func (r *MilvusSearchResult) ItemsCountByJob() map[int64]int32 {
	ret := make(map[int64]int32)
	for _, fitem := range r.FeatureItems {
		if cnt, ok := ret[fitem.JobID]; !ok {
			ret[fitem.JobID] = 1
		} else {
			ret[fitem.JobID] = cnt + 1
		}
	}

	return ret
}

/****************************** Milvus Data Store *************************************/
/**
 * @description: create default collection schema
 * @param {string} collection_name
 * @return {*}
 */
func (s *MilvusServer) createCollectionSchema(collection_name string) *entity.Schema {
	result := CollectionNameFormatRegexp.FindAllSubmatch([]byte(collection_name), -1)
	if len(result) == 0 {
		return nil
	}
	algo := string(result[0][1])

	var vec_dim string
	switch algo {
	case "ped_feat":
		vec_dim = "272"
	case "ped_attr":
		vec_dim = "48"
	case "face_feat":
		vec_dim = "256"
	case "bike_feat":
		vec_dim = "256"
	case "bike_attr":
		vec_dim = "48"
	case "vehicle_feat":
		vec_dim = "256"
	default:
		return nil
	}

	// create schema for collection
	schema := &entity.Schema{
		CollectionName: collection_name,
		Description:    "Collection for algorithm " + collection_name,
		AutoID:         false,
		Fields: []*entity.Field{
			{
				Name:        "feature_id",
				PrimaryKey:  true,
				AutoID:      false,
				DataType:    entity.FieldTypeInt64,
				Description: "primary key field",
			},
			{
				Name:        "job_id",
				PrimaryKey:  false,
				AutoID:      false,
				DataType:    entity.FieldTypeInt64,
				Description: "",
			},
			{
				Name:        "track_id",
				PrimaryKey:  false,
				AutoID:      false,
				DataType:    entity.FieldTypeInt64,
				Description: "",
			},
			{
				Name:        "timestamp_int64",
				PrimaryKey:  false,
				AutoID:      false,
				DataType:    entity.FieldTypeInt64,
				Description: "",
			},
			{
				Name:       "feature",
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

	return schema
}

/**
 * @description: check if milvus database is feasible for storig or searching
 * @param {context.Context} ctx
 * @param {*} collection_name
 * @param {string} partition_name
 * @param {bool} store : if this flag is set, it will create non-exist collection/partition
 * @param {*featurepb.CellFeature} cellFeature
 * @return {bool: milvus is feasible to further operation, error: err occurs during feasibility check}
 */
func (s *MilvusServer) checkMilvusFeasibility(ctx context.Context, collection_name string, partition_names []string, store bool) (bool, error) {
	collection_exists, ok := s.initCollection[collection_name]

	// collection not exists, return empty result/create collection
	if !collection_exists || !ok {
		if store {
			s.collMtx.Lock()
			defer s.collMtx.Unlock()
			if init, ok := s.initCollection[collection_name]; !ok || !init {
				schema := s.createCollectionSchema(collection_name)
				if schema == nil {
					log.Error("createCollectionSchema failed")
					err := errors.New("createCollectionSchema failed")
					return false, err
				}

				err := s.milvusClient.CreateCollection(ctx, schema, 2)
				if err != nil {
					log.Error("CreateCollection failed: ", err)
					return false, err
				}
				s.initCollection[collection_name] = true
			}
		} else {
			err := errors.New("collection " + collection_name + " not exists")
			return false, err
		}
	}

	// collection exists, proceed to check partitions
	for _, partition_name := range partition_names {
		// empty partition_name refers to '_default'
		if partition_name == "" {
			continue
		}

		partition_exists, ok := s.initPartition[collection_name+partition_name]
		if !partition_exists || !ok {
			if store {
				s.partMtx.Lock()
				defer s.partMtx.Unlock()
				if init, ok := s.initPartition[collection_name+partition_name]; !ok || !init {
					s.initPartition[collection_name+partition_name] = true
					err := s.milvusClient.CreatePartition(ctx, collection_name, partition_name)
					if err != nil {
						log.Error("checkMilvusSearchFeasibility failed: ", err)
						return false, err
					}
				}
			} else {
				err := errors.New("partition " + partition_name + " not exists")
				return false, err
			}
		}
	}

	return true, nil
}

/**
 * @description: create IVF_PQ index in specified collection
 * @param {string} collection_name
 * @return {*}
 * ! before calling this function, the collection should be guaranteed to exist cause it will not check
 */
// TODO: support multiple index creation and index parameters
func (s *MilvusServer) createVectorIndex(ctx context.Context, collection_name string) error {
	vecidx, err := entity.NewIndexIvfPQ(entity.L2, 2048, 16, 8)
	if err != nil {
		log.Errorf("milvusCreateVectorIndex failed: ", err)
		return err
	}

	err = s.milvusClient.CreateIndex(ctx, collection_name, "feature", vecidx, false)
	if err != nil {
		log.Errorf("milvusCreateVectorIndex failed: ", err)
		return err
	}

	return nil
}

/**
 * @description: insert a single vector into Milvus database
 * @param
 * ! before calling this function, the collection and partition should be guaranteed to exist cause it will not check
 */
func (s *MilvusServer) insertVectorByCols(ctx context.Context, collection_name string, partition_name string, cols []entity.Column) error {
	if cols == nil {
		return nil
	}

	// insert columns into collection
	_, err := s.milvusClient.Insert(ctx, collection_name, partition_name, cols...)
	if err != nil {
		log.Errorf("milvus insert failed: %v", err)
		return err
	}
	return nil
}

/**
 * @description: load the collections into memory
 * @param {context.Context} ctx
 * @param {string} collection_name
 * @param {[]string} partition_names, not used for now
 * @param {bool} async
 * @return {*}
// ! before calling this function, index must be built cause it will not check
*/
// TODO: when milvus supports collection/partitions loading in smaller granularity, modify this function
func (s *MilvusServer) milvusLoad(ctx context.Context, collection_name string, partition_names []string, async bool) (bool, error) {

	// check if current collection is loaded
	if loaded := s.collectionLoadMap[collection_name]; !loaded {
		// drop all other loaded collections first
		// for cname, ok := range s.collectionLoadMap {
		// 	if ok && cname != collection_name {
		// 		err := s.milvusClient.DropCollection(ctx, cname)
		// 		if err != nil {
		// 			log.Error("milvusLoad failed: ", err)
		// 			return false, err
		// 		}
		// 	}
		// }

		// load target collection
		err := s.milvusClient.LoadCollection(ctx, collection_name, async)
		if err != nil {
			log.Error("milvusLoad failed: ", err)
			return false, err
		}

		// mark as loaded
		s.collectionLoadMap[collection_name] = true
	}

	return true, nil
}

/****************************** Milvus API Functions *************************************/

/**
 * @description: initialize milvus connection and load metadata info
 * @return {}
 */
func (s *MilvusServer) Init() {
	// get milvus service address
	milvusIP := viper.GetString("milvus.addr")
	if milvusIP == "" {
		log.Panic("Empty Milvus Address")
	}
	milvusAddr := milvusIP + ":19530"

	// connect to the milvus server
	milvusClient, err := client.NewGrpcClient(context.Background(), milvusAddr)
	if err != nil {
		log.Panic("Milvus connect failed: %s", err.Error())
	}
	log.Println("Milvus Connected")

	// initialize fields
	s.milvusClient = milvusClient
	s.collectionIndexMap = make(map[string]bool)
	s.collectionLoadMap = make(map[string]bool)
	s.initCollection = make(map[string]bool)
	s.initPartition = make(map[string]bool)
	s.collMtx = &sync.Mutex{}
	s.partMtx = &sync.Mutex{}

	// initialize api info from milvus server
	// get all collection in Milvus
	collections, err := s.milvusClient.ListCollections(context.Background())
	if err != nil {
		log.Warn("Get collection list failed: ", err.Error())
		return
	}

	// describe collections
	for _, coll := range collections {
		// update init map
		s.initCollection[coll.Name] = true

		// update load map
		s.collectionLoadMap[coll.Name] = coll.Loaded

		// update index map
		indexProgress, err := s.milvusClient.GetIndexState(context.Background(), coll.Name, "feature")
		if err != nil {
			log.Warn("Get index state of ", coll.Name, " failed:", err.Error())
			continue
		}

		switch indexProgress {
		case 3:
			s.collectionIndexMap[coll.Name] = true
		default:
			s.collectionIndexMap[coll.Name] = false
		}

		// update partition map
		partitions, err := s.milvusClient.ShowPartitions(context.Background(), coll.Name)
		if err != nil {
			log.Warn("Get partitions of ", coll.Name, " failed:", err.Error())
			continue
		}

		if partitions == nil {
			continue
		} else {
			for _, partition := range partitions {
				s.initPartition[coll.Name+partition.Name] = true
			}
		}
	}
}

/**
 * @description: check collection by name
 * @param {context.Context} ctx
 * @param {string} collection_name
 * @return {*}
 */
func (s *MilvusServer) MilvusCheckCollection(ctx context.Context, collection_name string) (bool, error) {
	return s.milvusClient.HasCollection(ctx, collection_name)
}

/**
 * @description: check partition by name
 * @param {*} ctx
 * @param {*} collection_name
 * @param {*} partition_name
 * @return {*}
 */
func (s *MilvusServer) MilvusCheckPartition(ctx context.Context, collection_name, partition_name string) (bool, error) {
	return s.milvusClient.HasPartition(ctx, collection_name, partition_name)
}

/**
 * @description: perform milvus search
 * @param {context.Context} ctx
 * @param {MilvusSearchReq} req
 * @param {[]string} outputFields
 * @return {[]*MilvusSearchResult, error}
 */
func (s *MilvusServer) MilvusSearchFeatures(ctx context.Context, req MilvusSearchReq) ([]*MilvusSearchResult, error) {
	// check milvus feasibility
	feasible, err := s.checkMilvusFeasibility(ctx, req.CollName, req.PartitionNames, false)
	if err != nil || !feasible {
		log.Error("MilvusSearchFeatures droped: ", err)
		return nil, err
	}

	// check if index is built
	if indexed := s.collectionIndexMap[req.CollName]; !indexed {
		err := s.createVectorIndex(ctx, req.CollName)
		if err != nil {
			log.Error("MilvusSearchFeatures droped: ", err)
			return nil, err
		}
	}

	// ahead of searching, check if target collection/partition is loaded
	ok, err := s.milvusLoad(ctx, req.CollName, req.PartitionNames, false)
	if err != nil || !ok {
		log.Error("MilvusSearchFeatures droped: ", err)
		return nil, err
	}

	// prepare for search
	sp, err := entity.NewIndexIvfPQSearchParam(int(req.Nprobe))
	if err != nil {
		log.Error("MilvusSearchFeatures droped: ", err)
		return nil, err
	}

	vec := []entity.Vector{}
	for _, fea := range req.Feature {
		vec = append(vec, entity.FloatVector(fea))
	}

	// perform search
	start := time.Now()
	mrets, err := s.milvusClient.Search(
		ctx,
		req.CollName,
		req.PartitionNames,
		req.Expr,
		req.OutputFields,
		vec,
		req.VecName,
		req.MetricType,
		int(req.TopK),
		sp,
	)
	log.Info("Milvus Search Exec Time: [", time.Since(start).Milliseconds(), "ms]")

	// handle the search results
	if err != nil {
		log.Error("MilvusSearchFeatures failed: ", err)
		return nil, err
	}

	parseName := CollectionNameFormatRegexp.FindAllSubmatch([]byte(req.CollName), -1)
	if len(parseName) == 0 {
		return nil, nil
	}
	algoString := string(parseName[0][1])
	rawResults := milvusPackSearchResults(mrets, req.OutputFields, algoString, req.ScoreThreshold)

	// if ranking not needed
	if req.NoRanking {
		return rawResults, nil
	}

	// perform ranking
	// TODO: optimize ranking
	rankedResults, err := s.milvusRanking(ctx, rawResults, req)
	if err != nil {
		log.Error("MilvusSearchFeatures failed: ", err)
		return nil, err
	}

	return rankedResults, nil

	// test
	// qres, err := s.milvusRetriveVectors(ctx, &results, req, true)
	// for _, q := range qres {
	// 	log.Println("FeatureID: ", q.FeatureIndex)
	// 	log.Println("FeatureVCnum: ", len(q.FeatureItems))
	// }

	// return results, nil
}

/**
 * @description: Store Data into Milvus
 * @param {context.Context} ctx
 * @param {string} collection_name
 * @param {string} partition_name
 * @param {[]entity.Column} cols
 * @return {error}
 */
func (s *MilvusServer) MilvusStoreFeature(ctx context.Context, collection_name, partition_name string, cols []entity.Column) error {
	// check and prepare milvus database
	feasible, err := s.checkMilvusFeasibility(ctx, collection_name, []string{partition_name}, true)
	if err != nil || !feasible {
		log.Error("MilvusStoreFeature droped: ", err)
		return err
	}
	log.Println("Milvus Feasilbe to Store")

	// insert data
	err = s.insertVectorByCols(ctx, collection_name, partition_name, cols)
	if err != nil {
		log.Error("MilvusStoreFeature failed: ", err)
		return err
	}

	// if index is not built
	if indexed, ok := s.collectionIndexMap[collection_name]; !indexed || !ok {
		err = s.createVectorIndex(ctx, collection_name)
		if err != nil {
			log.Errorf("MilvusStoreFeature failed: ", err)
			return err
		}
		s.collectionIndexMap[collection_name] = true
	}

	// if collection is not loaded
	// if loaded := s.collectionLoadMap[collection_name]; !loaded {
	// 	err := s.milvusClient.LoadCollection(ctx, collection_name, false)
	// 	if err != nil {
	// 		log.Errorf("milvusStoreFeature: ", err)
	// 		return err
	// 	}
	// 	s.collectionLoadMap[collection_name] = true
	// }

	return nil
}

/**
 * @description: Query to retrive feature vectors from Milvus
 * @param {context.Context} ctx
 * @param {MilvusSearchReq} req
 * @return {*}
 */
func (s *MilvusServer) MilvusQuery(ctx context.Context, req MilvusSearchReq) ([]entity.Column, []MilvusQueryItem, error) {
	// check milvus feasibility
	feasible, err := s.checkMilvusFeasibility(ctx, req.CollName, req.PartitionNames, false)
	if err != nil || !feasible {
		log.Error("MilvusQuery droped: ", err)
		return nil, nil, err
	}

	// check if index is built
	if indexed, ok := s.collectionIndexMap[req.CollName]; !indexed || !ok {
		err := s.createVectorIndex(ctx, req.CollName)
		if err != nil {
			log.Warn("MilvusQuery droped: ", err)
			return nil, nil, err
		}
	}

	// ahead of searching, check if target collection/partition is loaded
	ok, err := s.milvusLoad(ctx, req.CollName, req.PartitionNames, false)
	if err != nil || !ok {
		log.Warn("MilvusQuery droped: ", err)
		return nil, nil, err
	}

	// perform query
	start := time.Now()
	cols, err := s.milvusClient.Query(
		ctx,
		req.CollName,
		req.PartitionNames,
		req.Expr,
		req.OutputFields,
	)
	log.Info("Milvus Query Exec Time: [", time.Since(start).Milliseconds(), "ms]")

	if err != nil {
		log.Error("MilvusQuery faield: ", err)
		return nil, nil, err
	}

	// parse returned result and pack into MilvusQueryResult
	var featureidCol *entity.ColumnInt64
	var featureCol *entity.ColumnFloatVector
	for _, col := range cols {
		switch col.Name() {
		case "feature_id":
			if c, ok := col.(*entity.ColumnInt64); ok {
				featureidCol = c
			}
		case "feature":
			if c, ok := col.(*entity.ColumnFloatVector); ok {
				featureCol = c
			}
		}
	}

	result := []MilvusQueryItem{}
	// log.Println("Query DEBUG::")
	// log.Println("\tfeatureidCol len::", featureidCol.Len())

	for i := 0; i < featureidCol.Len(); i++ {
		fid, _ := featureidCol.ValueByIdx(i)
		res := MilvusQueryItem{
			CollName:   req.CollName,
			FeatureID:  fid,
			FeatureVec: featureCol.Data()[i],
		}
		result = append(result, res)
	}

	return cols, result, nil
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

/**
 * @description: pack the Milvus client search results to []MilvusSearchResult
 * @param {[]client.SearchResult} Rets
 * @param {[]string} outputFields
 * @return {*}
 */
func milvusPackSearchResults(Rets []client.SearchResult, outputFields []string, algoString string, scoreThreshold float32) []*MilvusSearchResult {
	if Rets == nil {
		return nil
	}

	results := []*MilvusSearchResult{}
	collMap := make(map[string]*entity.ColumnInt64)

	for id, res := range Rets {
		// log.Println("Parse Fields::")
		for _, field := range res.Fields {

			// log.Println("\tFields::", field.Name())
			switch field.Name() {
			case "feature_id":
				if c, ok := field.(*entity.ColumnInt64); ok {
					collMap["feature_id"] = c
				}
			case "job_id":
				if c, ok := field.(*entity.ColumnInt64); ok {
					collMap["job_id"] = c
				}
			case "track_id":
				if c, ok := field.(*entity.ColumnInt64); ok {
					collMap["track_id"] = c
				}
			case "timestamp_int64":
				if c, ok := field.(*entity.ColumnInt64); ok {
					collMap["timestamp_int64"] = c
				}
			default:
				continue
			}
		}

		// construct search result
		result := &MilvusSearchResult{}

		// DEBUG
		// log.Println("DEBUG::")
		for i := 0; i < res.ResultCount; i++ {
			featureid, _ := collMap["feature_id"].ValueByIdx(i)
			jobid, _ := collMap["job_id"].ValueByIdx(i)
			timestamp, _ := collMap["timestamp_int64"].ValueByIdx(i)
			trackid, _ := collMap["track_id"].ValueByIdx(i)
			score := calcScore(res.Scores[i], algoString)

			if score > scoreThreshold {
				continue
			}

			// log.Println("\tResult::", i, " ", featureid, " ", res.Scores[i])
			// construct return item
			item := MilvusSearchItem{
				FeatureID: featureid,
				JobID:     jobid,
				TrackID:   trackid,
				Timestamp: timestamp,
				Score:     score,
			}
			result.FeatureItems = append(result.FeatureItems, item)
		}
		result.FeatureIndex = id
		result.TotalHits = 0 // ???
		results = append(results, result)
	}
	return results
}

// TODO: support more algorithm calculation
func calcScore(dis float32, algo string) float32 {
	var res float32
	switch algo {
	case "face_feat":
		res = dis
	case "ped_feat":
		newdisNormalization := 2 + 0.2*6.3442
		newDis := float64(dis) / newdisNormalization
		res = float32(math.Exp(float64(-1.0 * math.Acos(-1.0) * math.Pow(newDis, 2))))
	default:
		res = dis
	}
	return res
}
