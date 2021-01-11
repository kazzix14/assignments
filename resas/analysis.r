#install.packages("ensurer")
#install.packages("psych")

library(ensurer)
library(psych)

data <- read.csv("df.csv")

data$engineers_ratio <- data$engineers / data$population
data$patients_ratio <- data$patients_ratio / 100000
engineers_ratio <- data[,'engineers_ratio']
patients_ratio <- data[,'patients_ratio']

plot(data[,c('engineers_ratio', 'patients_ratio')])
hist(engineers_ratio)
hist(patients_ratio)

st_er <- shapiro.test(engineers_ratio)
st_pr <- shapiro.test(patients_ratio)

# 95%信頼度で正規分布ではない
ensure_that(st_er$p.value, . < 0.05) 
ensure_that(st_pr$p.value, . < 0.05) 

# pearsonは正規分布でないと使えない
# spearmanとkendallはピットマン相対効率は0.91なのでどっちでもよい
result_cor_spearman = cor.test(engineers_ratio, patients_ratio, method = "spearman")
# spearmanは順位をデータとしてpearsonをやっただけなのでこれはspearmanと一致する
# spearmanだと信頼区間を出してくれないので
result_cor_spearman2 = cor.test(rank(engineers_ratio), rank(patients_ratio), method = "pearson")
result_cor_kendall = cor.test(engineers_ratio, patients_ratio, method = "kendall")

# 95%信頼度で相関はない
# 帰無仮説: 相関はない
# p < 0.05 なので帰無仮説は棄却される
# 95%の信頼度で相関があると言える.
ensure_that(result_cor_spearman$p.value, . < 0.05)
ensure_that(result_cor_kendall$p.value, . < 0.05)

# 弱い負の相関がありそう
# 95%信頼区間も-0.5 ~ -0.05って感じなので負の相関ぽい
result_cor_spearman
result_cor_spearman2
result_cor_kendall

psych::pairs.panels(data[,c('patients_ratio','engineers_ratio')], method='spearman')

write.table(data[,c('patients_ratio','engineers_ratio')], file="df2.csv", sep=",")