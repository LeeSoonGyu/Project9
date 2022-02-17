library(tensorflow)
library(keras)
library(dplyr)
# install.packages('tfdatasets')
library(tfdatasets)
library(caret)
library(dplyr)
library(ggplot2)

# 보스턴 주택 가격 데이터셋
boston <- dataset_boston_housing()
c(train_data, train_labels) %<-% boston$train
c(test_data, test_labels) %<-% boston$test

paste0("Training entries: ", length(train_data), ", labels: ", length(train_labels))


column_names <- c('CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
                  'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT')

train_df <- train_data %>% 
  as_tibble(.name_repair = "minimal") %>% 
  setNames(column_names) %>% 
  mutate(label = train_labels)

test_df <- test_data %>% 
  as_tibble(.name_repair = "minimal") %>% 
  setNames(column_names) %>% 
  mutate(label = test_labels)

train_df
test_df
str(train_df)

train_df_scale <- transform(train_df,
                            CRIM = scale(train_df$CRIM, center = T, scale = T),
                            ZN = scale(train_df$ZN, center = T, scale = T),
                            INDUS = scale(train_df$INDUS, center = T, scale = T),
                            CHAS = scale(train_df$CHAS, center = T, scale = T),
                            NOX = scale(train_df$NOX, center = T, scale = T),
                            RM = scale(train_df$RM, center = T, scale = T),
                            AGE = scale(train_df$AGE, center = T, scale = T),
                            DIS = scale(train_df$DIS, center = T, scale = T),
                            RAD = scale(train_df$RAD, center = T, scale = T),
                            TAX = scale(train_df$TAX, center = T, scale = T),
                            PTRATIO = scale(train_df$PTRATIO, center = T, scale = T),
                            B = scale(train_df$B, center = T, scale = T),
                            LSTAT = scale(train_df$LSTAT, center = T, scale = T),
                            label = scale(train_df$label, center = T, scale = T))
train_df_scale


test_df_scale <- transform(test_df,
                            CRIM = scale(test_df$CRIM, center = T, scale = T),
                            ZN = scale(test_df$ZN, center = T, scale = T),
                            INDUS = scale(test_df$INDUS, center = T, scale = T),
                            CHAS = scale(test_df$CHAS, center = T, scale = T),
                            NOX = scale(test_df$NOX, center = T, scale = T),
                            RM = scale(test_df$RM, center = T, scale = T),
                            AGE = scale(test_df$AGE, center = T, scale = T),
                            DIS = scale(test_df$DIS, center = T, scale = T),
                            RAD = scale(test_df$RAD, center = T, scale = T),
                            TAX = scale(test_df$TAX, center = T, scale = T),
                            PTRATIO = scale(test_df$PTRATIO, center = T, scale = T),
                            B = scale(test_df$B, center = T, scale = T),
                            LSTAT = scale(test_df$LSTAT, center = T, scale = T),
                            label = scale(test_df$label, center = T, scale = T))
test_df_scale



model <- lm(label ~ ., data = train_df_scale)
summary(model)
model2 <- step(model)

summary(model2)

library('lmtest')
dwtest(model2)  # 더빈 왓슨 값의 p-value가 0.05이상이므로 독립성 있다.

plot(model2, which = 1)

attributes(model2)
res <- residuals(model)
shapiro.test(res) # shapiro의 p-value값은 0.05이하이므로 정규분포를 따르지 않는다.
par(mfrow = c(1, 1))
hist(res, freq = F)
qqnorm(res)

library(car)
sqrt(vif(model)) > 3.3  # 다중 공선성 문제 없음

# 예측
model2_pred <- predict(model2, newdata = test_df_scale)

# 예측과 실측값 비교
plot(model2_pred, test_labels_nor)
abline(a=0,b=1,col="blue",lty=1)

mean(abs(model2_pred - test_labels_nor))  # mae값 
# 3.318 * 1000 으로 3318달러의 오차수준으로 예상하고 있다.

qq <- test_labels - mean(test_labels)
test_labels_nor <- qq / sd(test_labels)
