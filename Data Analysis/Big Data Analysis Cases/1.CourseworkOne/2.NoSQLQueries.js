json    

// query 1: Calculate each sector's beta and find the sector which has the highest beta (or risk)
db.CourseworkOne.aggregate([ 
    {$group: {_id: "$StaticData.GICSSector", beta: {$avg: "$MarketData.Beta"}}}, 
    {$sort: {beta:-1} }
    ])


// query 2: we observe the beta of each subindustry in IT sector
db.CourseworkOne.aggregate([ 
    {$match: {"StaticData.GICSSector":/^Info./ }}, 
    {$group: {_id: "$StaticData.GICSSubIndustry", beta: {$avg: "$MarketData.Beta"}}}, 
    {$sort: {beta:-1} }
    ])
// the regex equals to 
// db.CourseworkOne.aggregate([ {$match: {"StaticData.GICSSector":{"$eq":"Information Technology"}}}, {$group: {_id: "$StaticData.GICSSubIndustry", beta: {$avg: "$MarketData.Beta"}}}, {$sort: {beta:-1} }])


