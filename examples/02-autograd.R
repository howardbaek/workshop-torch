library(torch)
library(dplyr)

# Computing derivatives with autograd

x <- torch_tensor(1, requires_grad = TRUE)
x$requires_grad

y <- 2 * x #dy/dx = 2

y$backward()
x$grad

x <- torch_tensor(100, requires_grad = TRUE)
y <- x^2 # dy/dx = 2*x^(2-1) = 2x
y$backward()
x$grad

x <- torch_tensor(1, requires_grad = TRUE)
y <- 2 * x^2 # dy/dx = 4x
y$backward()
x$grad


# Regression on mtcars

# first let's get the baseline with base R's lm
x <- mtcars %>% select(-mpg) %>% scale()
y <- scale(mtcars$mpg)

scale_mtcars <- mtcars %>% mutate(across(everything(), scale))
mod <- lm(mpg ~ ., data = scale_mtcars)
summary(mod)

mean(mod$residuals^2) # MSE from lm

# Now the same with torch

lr <- 0.001

W <- torch_randn(10, 1, requires_grad = TRUE)
b <- torch_zeros(1, 1, requires_grad = TRUE)

x_t <- torch_tensor(x)
y_t <- torch_tensor(y)

for (i in 1:50000) {
  y_hat <- x_t$mm(W) + b # forward pass
  loss <- torch_mean((y_t$view(c(-1, 1)) - y_hat)^2)
  
  loss$backward()
  
  with_no_grad({
    W$sub_(lr * W$grad)
    b$sub_(lr * b$grad)
    
    W$grad$zero_()
    b$grad$zero_()
  })
  
  if (i %% 1000 == 0)
    cat("Loss ", loss$item(), "\n")
}


coef(mod)
c(as.numeric(b), as.numeric(W))

# Exercise!
# Can you minimize the rosenbrock function using gradient descent?

rosenbrock <- function(x) {
  x1 <- x[1]
  x2 <- x[2]
  (a - x1)^2 + b * (x2 - x1^2)^2
}

# tip: https://skeydan.github.io/Deep-Learning-and-Scientific-Computing-with-R-torch/optim_1.html
