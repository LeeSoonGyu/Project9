#빅데이터A 2차 한호종
#keras

rm(list = ls())
.rs.restartR()

#3. ‘Deep Learning with R’ 교재 내 5.2 Using convnets with small datasets 섹션에서 개와
#고양이의 이미지 분류 문제에서 다음 조건을 고려하여 실행하시오.
install.packages("keras")
install.packages("tensorflow")
library(tensorflow)
library(keras)
install_keras()
tensorflow::install_tensorflow()
install_tensorflow(version = "gpu")
install_keras(tensorflow = "gpu")

#install_keras(tensorflow = "gpu")
library(dplyr)
##교육, 검증 및 테스트 디렉토리에 이미지 복사

setwd('C:/Rwork/dogs-vs-cats')

original_dataset_dir <- "C:/Rwork/dogs-vs-cats/train"

base_dir <- "C:/Rwork/dogs-vs-cats/cat_and_dog_small"

## 학습,검증,테스트 경로 생성
train_dir <- file.path(base_dir, "train")
validation_dir <- file.path(base_dir, "validation")
test_dir <- file.path(base_dir, "test")
train_cats_dir <- file.path(train_dir, "cats")
train_dogs_dir <- file.path(train_dir, "dogs")
validation_cats_dir <- file.path(validation_dir, "cats")
validation_dogs_dir <- file.path(validation_dir, "dogs")
test_cats_dir <- file.path(test_dir, "cats")
test_dogs_dir <- file.path(test_dir, "dogs")

## 파일 생성 뭉치기
a <- c(train_dir,test_dir,validation_dir,train_cats_dir,train_dogs_dir,
       validation_cats_dir,validation_dogs_dir,test_cats_dir,test_dogs_dir)
lapply(a,dir.create)

## 설정값 이미지파일 각각 분류

fnames <- paste0(2001:3000,".jpg")
file.copy(file.path(original_dataset_dir, "Cat", fnames), 
          file.path(train_cats_dir))
fnames <- paste0(3001:3500, ".jpg")
file.copy(file.path(original_dataset_dir, "Cat", fnames), 
          file.path(validation_cats_dir))
fnames <- paste0(3501:4000, ".jpg")
file.copy(file.path(original_dataset_dir, "Cat", fnames),
          file.path(test_cats_dir))

fnames <- paste0(4001:5000, ".jpg")
file.copy(file.path(original_dataset_dir, "Dog", fnames),
          file.path(train_dogs_dir))
fnames <- paste0(5001:5500, ".jpg")
file.copy(file.path(original_dataset_dir, "Dog", fnames),
          file.path(validation_dogs_dir)) 
fnames <- paste0(5501:6000, ".jpg")
file.copy(file.path(original_dataset_dir, "Dog", fnames),
          file.path(test_dogs_dir))

## 분류 데이터 개수 확인

cat("total training cat images:", length(list.files(train_cats_dir)), "\n")
cat("total training dog images:", length(list.files(train_dogs_dir)), "\n")
cat("total validation cat images:", length(list.files(validation_cats_dir)), "\n")
cat("total validation dog images:", length(list.files(validation_dogs_dir)), "\n")
cat("total test cat images:", length(list.files(test_cats_dir)), "\n")
cat("total test dog images:", length(list.files(test_dogs_dir)), "\n")

## model 파라미터 값 설정 

model <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(4, 4), activation = "relu",
                input_shape = c(150, 150, 3)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(4, 4), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128, kernel_size = c(4, 4), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128, kernel_size = c(4, 4), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_flatten() %>% 
  layer_dense(units = 512, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

summary(model)

## 학습을 위한 모델 구성

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(learning_rate = 1e-4),
  metrics = c("acc")
)

## 이미지 사이즈 조정 및 데이터 타입 변경

train_datagen <- image_data_generator(rescale = 1/255)
validation_datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(
  train_dir,
  train_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)

## 배치 생성기를 사용하여 모델 피팅

batch <- generator_next(train_generator)
str(batch)

history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 20,
  epochs = 10,
  validation_data = validation_generator,
  validation_steps = 10
)

## 모델 저장

#model %>% save_model_hdf5("cats_and_dogs_small_1.h5")

## 시각화

plot(history)


## 데이터 보강 구상 설정

datagen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)

## 무작위로 보강된 훈련 이미지 표시

fnames <- list.files(train_cats_dir, full.names = TRUE)
img_path <- fnames[[3]]

img <- image_load(img_path, target_size = c(150, 150))
img_array <- image_to_array(img)
img_array <- array_reshape(img_array, c(1, 150, 150, 3))

augmentation_generator <- flow_images_from_data(
  img_array,
  generator = datagen, 
  batch_size = 1
)

op <- par(mfrow = c(2, 2), pty = "s", mar = c(1, 0, 1, 0))
for (i in 1:4) {
  batch <- generator_next(augmentation_generator)
  plot(as.raster(batch[1,,,]))
}
par(op)

## 드롭아웃을 포함하는 새 convnet 정의

model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(150, 150, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")
model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4),
  metrics = c("acc")
)

## 데이터 증대 생성기를 사용하여 convnet 훈련

datagen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE
)
test_datagen <- image_data_generator(rescale = 1/255)
train_generator <- flow_images_from_directory(
  train_dir, 
  datagen, 
  target_size = c(150, 150), 
  batch_size = 32, 
  class_mode = "binary"
)
validation_generator <- flow_images_from_directory(
  validation_dir,
  test_datagen,
  target_size = c(150, 150),
  batch_size = 32,
  class_mode = "binary"
)
history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 20,
  epochs = 10,
  validation_data = validation_generator,
  validation_steps = 10
)

## 모델 저장 밑 시각화

#model %>% save_model_hdf5("cats_and_dogs_small_2.h5")

plot(history)
