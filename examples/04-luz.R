library(torch)
library(luz)
library(dplyr)

x <- mtcars %>% select(-mpg) %>% scale()
y <- scale(mtcars$mpg)

lr <- 0.001

x_t <- torch_tensor(x)
y_t <- torch_tensor(y)

fitted <- nn_linear %>% 
  setup(
    optimizer = optim_sgd,
    loss = function(y_hat, y) {
      torch_mean((y$view(c(-1, 1)) - y_hat)^2)
    }
  ) %>% 
  set_opt_hparams(lr = 0.001) %>% 
  set_hparams(in_features = 10, out_features = 1) %>% 
  fit(
    list(x_t, y_t),
    epochs = 100, # epoch: how many times you pass through your data
    dataloader_options = list(batch_size = 32) # batch size: small portion of your data
  )

luz_save(fitted, "model.luz")

predict(fitted, x_t)
plot(fitted) # get a view of your loss over epoch

# luz will automatically detect the fastest device (GPU, CPU, MPS) by default and run computation on the device
