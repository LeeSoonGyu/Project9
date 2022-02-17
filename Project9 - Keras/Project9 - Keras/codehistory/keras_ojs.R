# Project9 =================================================================

  #환경설정
rm(list=ls())
setwd('c:/rwork/')

  #라이브러리
#install.packages("tfdatasets")
library(tfdatasets)
#install.packages("reticulate")
#install.packages("tensorflow")
library(tensorflow)
#install.packages("keras")
library(keras)

tensorflow::install_tensorflow()
install_keras()
#tensorflow::tf_config()

#install.packages("dplyr")
library(dplyr)
#install.packages("lmtest")
library(lmtest)
#install.packages('car')
library(car)

use_condaenv("r-tensorflow")


# Keras 다중회귀분석

  #데이터셋 설명)
  #CRIM - 도시별 1인당 범죄율
  #ZN - 25,000평방피트 이상의 용지에 대해 구역이 지정된 주거용 토지의 비율.
  #INDUS - 도시당 비소매 사업 면적의 비율
  #CHAS - Charles River 더미 변수(강의 경계지역일 경우 1, 그렇지 않을 경우 0)
  #NOX - 산화질소 농도(1천만 개당 부품)
  #RM - 주거지당 평균 객실 수
  #AGE - 1940년 이전에 제작된 자가 사용 장치의 비율
  #DIS - 보스턴 고용 센터 5곳까지의 가중 거리
  #RAD - 방사형 고속도로 접근성 지수
  #세금 - $10,000당 전체 가치 재산세율
  #PTRATIO - 읍별 학생-교사 비율
  #B - 1000(Bk - 0.63)^2 여기서 Bk는 도시별 흑인의 비율
  #LSTAT - 모집단의 낮은 상태 %
  #MEDV - 1,000달러 단위의 자가 주택 중위수 값

  #목표: 산화질소의 수준을 예측하고 분석한다

# 1.데이터셋 준비
boston_housing <- dataset_boston_housing(test_split = 0.3,
                                         seed = 1234)
  # 테스트 셋의 default 비율은 0.2이다
str(boston_housing)
c(train_data, train_labels) %<-% boston_housing$train
c(test_data, test_labels) %<-% boston_housing$test

column_names <- c('CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
                  'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT')

train_df <- train_data %>% as_tibble(.name_repair = "minimal") %>%
  setNames(column_names) %>% mutate(label = train_labels)

test_df <- test_data %>% as_tibble(.name_repair = "minimal") %>%
  setNames(column_names) %>% mutate(label = test_labels)

spec <- feature_spec(train_df, label ~ . ) %>%
  step_numeric_column(all_numeric(), normalizer_fn = scaler_standard()) %>%
  fit()

spec

layer <- layer_dense_features(feature_columns = dense_features(spec), dtype = tf$float32)

layer(train_df)

# 2.다중회귀분석의 조건 확인
## 1번모델
### 1)모델생성 및 잔차분석
fom <- NOX ~ ZN + INDUS + CHAS + AGE
model <- lm(formula = fom, data = train_df)
model

### 2)독립성 검정
dwtest(model)
  #DW값이 1~3 안쪽이므로 독립성 문제없음

### 3)등분산성 검정
plot(model, which = 1)

### 4)잔차의 정규성 검정
res <- residuals(model)
shapiro.test(res)
hist(res, freq = F)
qqnorm(res)
  # p-value 값이 0.05 이하이며,
  # Q-Q 그래프에도 45도에 가깝긴 하나 이상치들이 있는 것으로 보이므로,
  # 정규성을 만족하지 못하는 것으로 판단됨

boxplot(train_df$ZN)
summary(train_df$ZN)

plot(train_df$CHAS)
plot(train_df$INDUS)
plot(train_df$AGE)
# 확인결과 CHAS는 범주형 변수이므로 제외해야 한다

### 5)다중 공선성 검정
vif(model) < 5
  # 분산팽창요인이 모두 10 이하이므로 다중 공선성에는 문제없음
cor(train_df[,c(2,3,4,7)], method = 'pearson')

### 6)모델 평가
summary(model)
  # 이미 정규성에 문제가 있으므로 다음 모델을 생성한다



## 2번모델
### 1)모델생성 및 잔차분석
train_df3 <- subset(train_df, ZN >= 20 & ZN <= 80)#등분산성에 문제 생김
fom2 = NOX ~ INDUS + AGE
model2 <- lm(formula = fom2, data = train_df)
  # 첫번째 모델에서 영향력이 가장 적었던 ZN 변수를 제외하였다
  # CHAS 변수는 범주형이므로 다중회귀분석에서 제외해야한다
summary(model2)

### 2)독립성 검정
dwtest(model2)

### 3)등분산성 검정
plot(model2, which = 1)

### 4)잔차의 정규성 검정
res2 <- residuals(model2)
shapiro.test(res2)
hist(res2, freq = F)
qqnorm(res2)
  #p-value가 0.05 이하이므로 귀무가설이 기각되어 잔차의 정규성에 문제가 있음을 확인 

### 5)다중 공선성 검정
vif(model2) < 5

### 6)모델 평가
summary(model2)



## 3번모델


# Using convnets with small datasets
# 