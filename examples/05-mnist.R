library(torch)
library(luz)
library(torchvision)

# create a dataset --------------------------------------------------------

root <- "./datasets/mnist"

# transform() 
transform <- function(x) {
  x %>% 
    transform_to_tensor() %>% 
    torch_flatten()
}

train_ds <- mnist_dataset(root, transform = transform, download = TRUE)
test_ds <- mnist_dataset(root, transform = transform, train = FALSE)


# Define the neural net ---------------------------------------------------

net <- nn_module(
  "MLP", # multilayer perceptron (multiple linear models chained together separated by activation function)
  initialize = function(in_features, out_features) {
    self$linear1 <- nn_linear(in_features, 512)
    self$linear2 <- nn_linear(512, 256)
    self$linear3 <- nn_linear(256, out_features)
    self$relu <- nn_relu()
  },
  forward = function(x) {
    x %>% 
      self$linear1() %>% 
      self$relu() %>% 
      self$linear2() %>% 
      self$relu() %>% 
      self$linear3()
  }
)

# fit with luz ------------------------------------------------------------

fitted <- net %>% 
  setup(
    loss = nnf_cross_entropy,
    optim = optim_adam,
    metrics = list(
      luz_metric_accuracy()
    )
  ) %>% 
  set_hparams(in_features = 784, out_features = 10) %>% 
  set_opt_hparams(lr = 1e-3) %>% 
  fit(train_ds, valid_data = 0.2, epochs = 2)


fitted %>% 
  evaluate(test_ds)
  

# Exercise:

# Can you improve this model with techniques learned in Chapter 15 of Dl with R book?
# https://skeydan.github.io/Deep-Learning-and-Scientific-Computing-with-R-torch/image_classification_1.html#classification-on-tiny-imagenet



