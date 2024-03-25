library(torch)
library(dplyr)

# if we'd like to modularize our code using just R tools, we probably would
# do something like this;

linear <- nn_module(
  "linear",
  initialize = function() {
    self$W <- nn_parameter(torch_randn(10, 1))
    self$b <- nn_parameter(torch_zeros(1, 1))
  },
  forward = function(x) {
    x$mm(self$W) + self$b
  }
)

model <- linear() # essentially calling the initialize() function inside linear
# model() is a function that calls forward()
model(torch_randn(10, 10))

# The optimizer is just an R object with a consistent API
opt <- optim_sgd(model$parameters, lr = 0.001)

# train on mtcars ------

x <- mtcars %>% select(-mpg) %>% scale()
y <- scale(mtcars$mpg)

lr <- 0.001

x_t <- torch_tensor(x)
y_t <- torch_tensor(y)


for (i in 1:50000) {
  
  # we use a function here
  y_hat <- model(x_t)
  
  loss <- torch_mean((y_t$view(c(-1, 1)) - y_hat)^2)
  
  loss$backward()
  
  # we can gradient updates in a for loop
  opt$step()
  opt$zero_grad()
  
  if (i %% 1000 == 0)
    cat("Loss ", loss$item(), "\n")
}
