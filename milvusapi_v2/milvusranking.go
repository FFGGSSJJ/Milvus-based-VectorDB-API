package milvusapi

import (
	"context"
	"math"
	"sort"
	"strconv"
	"time"

	log "github.com/sirupsen/logrus"
)

// TODO:perform ranking for returned results
func (s *MilvusServer) milvusRanking(ctx context.Context, results []*MilvusSearchResult, req MilvusSearchReq) ([]*MilvusSearchResult, error) {
	start := time.Now()
	// perform query and re-calculate score
	sres, _, err := s.vectorsCalcRanking(ctx, results, req)
	if err != nil {
		log.Error("milvusRanking failed: ", err)
		return nil, err
	}

	// sort by the calculated score
	for _, res := range sres {
		sort.Slice(res.FeatureItems, func(i, j int) bool {
			return res.FeatureItems[i].Score < res.FeatureItems[j].Score
		})
	}
	log.Info("Overall Ranking Exec Time: [", time.Since(start).Milliseconds(), "ms]")
	return sres, nil
}

func (s *MilvusServer) vectorsCalcRanking(ctx context.Context, results []*MilvusSearchResult, req MilvusSearchReq) ([]*MilvusSearchResult, []MilvusQueryResult, error) {
	qRets := []MilvusQueryResult{}
	sRets := []*MilvusSearchResult{}
	result := CollectionNameFormatRegexp.FindAllSubmatch([]byte(req.CollName), -1)
	if len(result) == 0 {
		return nil, nil, nil
	}
	algo := string(result[0][1])

	// qpool := workerpool.New(8)
	// mtx := &sync.Mutex{}
	for id, res := range results {
		qitems := []MilvusQueryItem{}

		// pack req
		qreqs := balanceQueryReq(250, res, req)
		// retrive QueryItems
		for _, qreq := range qreqs {
			_, qitem, err := s.MilvusQuery(ctx, qreq)
			if err != nil {
				log.Error("milvusRetriveVectors failed: ", err)
				return nil, nil, err
			}

			qitems = append(qitems, qitem...)
		}

		// pack to QueryResult
		qret := MilvusQueryResult{
			FeatureIndex: id,
			FeatureItems: qitems,
			DistMap:      make(map[int64]float64),
		}
		sRet := &MilvusSearchResult{
			FeatureIndex: res.FeatureIndex,
		}

		// update score with accurate ranking
		for _, qitem := range qitems {
			dist := calcRankingL2Dis(req.Feature[id], qitem.FeatureVec, len(req.Feature[id]))
			qret.DistMap[qitem.FeatureID] = dist
		}
		for _, sitem := range res.FeatureItems {
			// log.Println("Score before rank::", sitem.Score)
			rankedItem := MilvusSearchItem{
				FeatureID: sitem.FeatureID,
				JobID:     sitem.JobID,
				TrackID:   sitem.TrackID,
				Timestamp: sitem.Timestamp,
				Score:     calcScore(float32(qret.DistMap[sitem.FeatureID]), algo),
			}
			// log.Println("Score after rank::", rankedItem.Score)
			sRet.FeatureItems = append(sRet.FeatureItems, rankedItem)
		}

		// append to results
		qRets = append(qRets, qret)
		sRets = append(sRets, sRet)
	}

	return sRets, qRets, nil
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
			OutputFields:   []string{"feature_id", "feature"},
		}
		featIDs := "["

		for j := i * int(burst); j < (i+1)*int(burst); j++ {
			featIDs = featIDs + strconv.Itoa(int(result.FeatureItems[j].FeatureID))
			if j < (i+1)*int(burst)-1 {
				featIDs = featIDs + ", "
			} else {
				featIDs = featIDs + "]"
			}
		}
		r.Expr = "feature_id in " + featIDs
		reqs = append(reqs, r)
	}

	// pack offset req
	if offset > 0 {
		r := MilvusSearchReq{
			CollName:       req.CollName,
			PartitionNames: req.PartitionNames,
			Expr:           "",
			OutputFields:   []string{"feature_id", "feature"},
		}
		featIDs := "["

		for i := bnum * int(burst); i < bnum*int(burst)+offset; i++ {
			featIDs = featIDs + strconv.Itoa(int(result.FeatureItems[i].FeatureID))
			if i < bnum*int(burst)+offset-1 {
				featIDs = featIDs + ", "
			} else {
				featIDs = featIDs + "]"
			}
		}
		r.Expr = "feature_id in " + featIDs
		reqs = append(reqs, r)
	}

	return reqs
}

// TODO: remove and use C embedded function
func calcRankingL2Dis(src, dst []float32, dim int) float64 {
	var sqsum float64
	sqsum = 0
	for i := 0; i < dim; i++ {
		sqsum = sqsum + float64((src[i]-dst[i])*(src[i]-dst[i]))
	}
	return math.Sqrt(sqsum)
}
