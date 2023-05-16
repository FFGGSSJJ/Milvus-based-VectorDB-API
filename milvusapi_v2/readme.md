# Milvus APIv2


## Description
- Milvus API v2 is developed for the usage in my intern project. It mainly provides the ability to store/search/query vectors using Milvus. It also supports ranking after search performed. 

## Links
- Milvus Update Plan： https://wiki.lfaidata.foundation/display/MIL/Feature+plans
- Milvus Official Doc：https://milvus.io/docs
- Milvus Tech Explore：https://zhuanlan.zhihu.com/p/517553501
- Personal Note(en)：https://ffggssjj.notion.site/Milvus-4a5d00b2256249d4aea2c24c458b5289

## Milvus API Intro
- https://ffggssjj.notion.site/Milvus-in-Polaris-17d8e346445f40719e2903c1f14ed1b2


## To do in milvusapi/
- `milvusapi.go: 32`: struct `MilvusServer` contains `initCollection，initPartition，collMtx，partMtx`. 主要用于处理特征存储并发的情况
    - 目前的设计是在函数`checkMilvusFeasibility()`中根据`initCollection，initPartition`判断是否创建，再使用mtx保证仅有一个协程进行collection/partition的创建。可能需要进行优化。
- `milvusapi.go: 319`: the function `milvusLoad()` should be modified once Milvus supports partitions load one by one. 
    - 由于当前Milvus版本(v2.2.8)不支持partition load one by one，目前**实时摘要**以collection为最小单位进行create/load。版本支持后需要更改collection/partition创建规则和该函数
- `milvusapi.go: calcScore()`: 目前只支持计算形体特征(***ped_feat***)的分数
- `milvusapi.go: createCollectionSchema()`: 目前Go SDK不支持创建**scalar index**，需要在后续版本支持后手动创建
- `milvusapi.go: createCollectionSchema()`: 该函数目前通过算法名称得到对应向量长度，目前缺少了几个算法的向量长度
- `milvusapi.go`: 目前没有collection/partition memory release的函数
- `milvusranking.go: vectorsCalcRanking()`: 该函数用于检索查询得到的特征向量，通过`balanceQueryReq()`函数生成检索请求，目前没有实现并发
- `milvusranking.go: calcRankingL2Dis()`: 该函数用于计算精排距离，目前使用**Golang Math**计算，需要进行优化
- `milvusranking.go`: 精排整体可能需要进行优化


## To do in analysislib/
- `server.go: convMilvusRes2AnalysisRes()`: 该函数的目的是将Milvus查询结果转化为SearchAnalysisResult。目前的问题是SearchConditionIndex与实际索引对应不上。
    - 原因是在进行`Search()`时，我会通过`makeMilvusSearchReqs()`生成MilvusSearchReq，该函数会将collection_name相同的查询请求包装进同一个MilvusSearchReq，导致索引不准确
- `server.go: Search()`: 目前`SearchRequest`中的PageSize, PageNumber, JobOrDevice没有被利用
- `server.go: Search()`: 目前`SearchResultSummary.ScanCount`没有确定, see TBD
- `server.go: Search()`: 函数整体可能需要优化
    - 该函数的整体思路：通过`SearchRequest.AnalysisCellNames`得到`feature_id` -> 通过`feature_id`进行Milvus Query获得特征向量 -> 通过检索得到的特征向量调用`MilvusSearchFeatures`进行相似向量查询 -> 生成`SearchResultSummary`返回 -> 通过`convMilvusRes2AnalysisRes`获得具体信息存入redis缓存
- `server.go: filterSnapShotResults()`: 该函数为`GetSearchSnapshot`服务，主要用于根据具体查询条件过滤搜索结果（搜索结果从redis中获得），目前过滤部分写的比较粗糙

