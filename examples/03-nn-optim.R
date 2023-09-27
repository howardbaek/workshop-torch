library(torch)

# if we'd like to modularize our code using just R tools, we probably would
# do something like this;

linear_model <- function(x, state) {
  x$mm(state$W) + state$b
}

state <- list(
  W = torch_randn(10, 1, requires_grad = TRUE),
  b = torch_zeros(1, 1, requires_grad = TRUE)
)

# train on mtcars ------

x <- mtcars %>% select(-mpg) %>% scale()
y <- scale(mtcars$mpg)

lr <- 0.001

x_t <- torch_tensor(x)
y_t <- torch_tensor(y)


for (i in 1:50000) {
  
  # we use a function here
  y_hat <- linear_model(x_t, state)
  
  loss <- torch_mean((y_t$view(c(-1, 1)) - y_hat)^2)
  
  loss$backward()
  
  # we can gradient updates in a for loop
  with_no_grad({
    for(param in state) {
      param$sub_(lr * param$grad)
      param$grad$zero_()
    }
  })
  
  if (i %% 1000 == 0)
    cat("Loss ", loss$item(), "\n")
}
