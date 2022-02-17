# 주택 가격 예측 : 회귀 문제
import pandas as pd
import numpy as np


# 보스턴 주택 가격 데이터셋
from keras.datasets import boston_housing
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

train_data.shape
test_data.shape

train_targets
# 타깃은 주택의 중간 가격으로 천 달러 단위입니다.
# 1만 달러에서 5만 달러 사이입니다

# 순서 섞기
order = np.argsort(np.random.random(train_targets.shape))
train_data = train_data[order]
train_targets = train_targets[order]
# 학습을 할 때, 비슷한 데이터들을 연속해서 학습하게 되면 편항이 된다.
# 따라서 학습 데이터들을 적절하게 섞어주는 것이 필요함.

# 정규화
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std
# 서로 다른 범위를 갖고 있다면, 직접적인 비교가 어렵기 때문에 이를 동일한 범위를 갖도록 해주는 작업.


# 모델생성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))
# 입력 레이어, 히든 레이어, 출력 레이어 각 1개씩 전결합 (Fully-Connected) 레이어로 만들었다.
# 활성화 함수로는 ReLU를 사용했다.


# compiling
from tensorflow.keras.optimizers import Adam

model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['mae'])

# 손실 함수로는 MSE (Mean Square Error) 함수를 사용했고, 최적화 함수로는 학습률 0.001의 Adam을 사용했다.
# 평가 지표로는 MAE (Mean Absolute Error)를 사용했다.

# Training
from keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=20)
history = model.fit(train_data, train_targets, epochs=500, validation_split=0.2, callbacks=[early_stop])
# EarlyStopping은 지정한 epoch만큼 반복하는 동안 학습 오차에 개선이 없다면 자동으로 학습을 종료함.
# val_loss를 모니터링하여 20번의 epoch동안 개선이 없다면 종료하게 된다.
# 이것을 fit 메소드에 넘겨주어 학습을 하는 동안 사용을 하게 된다.


# 결과 시각화
import matplotlib.pyplot as plt
plt.plot(history.history['mae'], label='train mae')
plt.plot(history.history['val_mae'], label='val mae')
plt.xlabel('epoch')
plt.ylabel('mae [$1,000]')
plt.legend(loc='best')
plt.ylim([0, 5])
plt.show()


# Evaluation
test_loss, test_mae = model.evaluate(test_data, test_targets)
# mae(평균절대오차)가 3.3433 이므로 평균적으로 2,837 달러 정도의 오차 범위 내에서 예측하고 있다.

# Prediction
print(np.round(test_targets[:10]))

test_predictions = model.predict(test_data).flatten()
print(np.round(test_predictions[:10]))
# 실제값과 예측값의 비교이다.

# 비교 시각화
plt.scatter(test_targets, test_predictions)
plt.xlabel('test_targets')
plt.ylabel('test_predictions')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
# 몇몇 값을 제외하면 모델이 예측을 꽤 잘했다고 판단함.