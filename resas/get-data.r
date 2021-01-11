#install.packages("httr")
#install.packages("listviewer")

library(httr)
library(listviewer)

pref_codes = 1:47

engineers = matrix(nrow=47, ncol=1)
patients = matrix(nrow=47, ncol=1)
populations = matrix(nrow=47, ncol=1)

response <- GET(
  url = "https://opendata.resas-portal.go.jp/api/v1/medicalWelfare/medicalAnalysis/chart",
  query = list(
    year = 2014,
    dispType = 2, #  人口10万人あたりで表示する
    sort = 2, # コード順ソート
    matter1 = 1, # 医療需要
    matter2 = 102, # 病院の推計入院患者数（傷病分類別）
    prefCode = 1 
  ),
  add_headers("X-API-KEY" = "uIBywVT0BE2cpZTaMBDoA2G4howfbdKgN7T6Jytx")
)

res <- content(response)
patients[,1] <- sapply(res$result$data, function(data) { data$value })

for(pref_code in pref_codes) {
  print(pref_code)

  response <- GET(
    url = "https://opendata.resas-portal.go.jp/api/v1/municipality/employee/perYear",
    query = list(
      prefCode = pref_code,
      cityCode = '-',
      sicCode = 'G',
      simcCode ='39' 
    ),
    add_headers("X-API-KEY" = "uIBywVT0BE2cpZTaMBDoA2G4howfbdKgN7T6Jytx")
  )

  res = content(response)
  engineers[pref_code, 1] <- res$result$data[[3]]$value # 2014年のデータ

  # population
  response <- GET(
    url = "https://opendata.resas-portal.go.jp/api/v1/population/composition/perYear",
    query = list(
      prefCode = pref_code,
      cityCode = '-'
    ),
    add_headers("X-API-KEY" = "uIBywVT0BE2cpZTaMBDoA2G4howfbdKgN7T6Jytx")
  )

  res = content(response)
  populations[pref_code, 1] <- res$result$data[[1]]$data[[8]]$value # 2015
}

df = data.frame(
  populations = populations,
  engineers = engineers,
  patients_ratio = patients
)

write.table(df, file="df.csv", sep=",")