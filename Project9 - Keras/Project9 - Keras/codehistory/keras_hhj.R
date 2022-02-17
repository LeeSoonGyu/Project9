#빅데이터A 2차 한호종
#keras

#1. keras package 내 Boston Housing Price 데이터셋과 함수를 이용 다중회귀분석을 수행하시오.

#install.packages("keras")
library(keras)
#install_keras()

#install.packages("tfdatasets")
library(tfdatasets)

#pkgbuild::check_build_tools(debug = TRUE)
#devtools::install_github("rstudio/reticulate")
#install.packages("tensorflow")
#install.packages("reticulate")

#데이터셋 로딩
boston_housing <- dataset_boston_housing()

c(train_data, train_labels) %<-% boston_housing$train
c(test_data, test_labels) %<-% boston_housing$test

library(dplyr)

column_names <- c('CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
                  'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT')

train_df <- train_data %>% as_tibble(.name_repair = "minimal") %>%
  setNames(column_names) %>% mutate(label = train_labels)

test_df <- test_data %>% as_tibble(.name_repair = "minimal") %>%
  setNames(column_names) %>% mutate(label = test_labels)

#기능 정규화
spec <- feature_spec(train_df, label ~ . ) %>% step_numeric_column(all_numeric(), normalizer_fn = scaler_standard()) %>% fit()

spec

layer <- layer_dense_features(feature_columns = dense_features(spec), dtype = tf$float32)

layer(train_df)

####
####

#회귀분석의 기본 가정 충족으로 회구분석 진행
#1 회귀모델 생성
model <- lm(formula = label ~ ., data = train_df)
model
#2-1 독립성 검정 - 더빈 왓슨 값으로 확인
#install.packages('lmtest')
library(lmtest)
dwtest(model)
#2-2 등분산성 검정 - 잔차와 적합값의 분포
plot(model, which = 1)
#2-3 잔차의 정규성 검정
attributes(model)
res <- residuals(model)
shapiro.test(res)
par(mfrow = c(1,2))
hist(res, freq = F)
summary(model)
qqnorm(res)
#3 다중 공선성 검사
#install.packages('car')
library(car)
sqrt(vif(model)) > 2
#4 회귀모델 생성과 평가
formula = label ~ NOX + DIS + RAD + TAX
model2 <- lm(formula = formula, data = train_df)
summary(model2)
