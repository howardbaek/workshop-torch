library(torch)

# A matrix with 100 rows and 10 columns
m <- matrix(runif(1000), nrow = 100)
# An array with 3 dimensions
a <- array(runif(10000), dim = c(100, 10, 10))

# Create torch tensors from R objects
t_m <- torch_tensor(m, dtype = "double") # CPUDoubleType{100, 10}: A tensor that lives on the CPU and a floating type tensor with dimensions [100, 10]
t_a <- torch_tensor(a)
# In general, use 'dtype = "float"' unless you need double precision!

# Convert torch tensors back to R
as.array(t_m)
as.matrix(t_m)

# Attributes of a tensor
t_m$device # 'device' : the location where your tensor is stored (e.g. CPU or GPU)
# and device (GPU/CPU) is where the computation is happening. 
t_m$dtype # 'dtype' : data type of the tensor
t_m$shape # 'shape' : equivalent to dim(t_m) in R
t_m$requires_grad # 'requires_grad' : related to autograd system

# Methods: torch tensors are R6 objects in R and these methods are functions 
# that can access the tensor. They are all mathematical operations you can do arrays.
(t_m -1)$abs()
t_m$add(1)
t_m$sum()
t_m$min()

# in-place methods end in `_`: modify the tensor in-place
t_m$add_(1)
t_m # t_m is modified in place, no copy is made
# This exists mostly for performance reasons (to avoid expensive operations)

# Built-in functions (similar to 'sum()', 'mean()' in R)
torch_mm(t_m, t_m$t())
torch_sum(t_m)
torch_mean(t_m)
# Usually, there is a 1:1 mapping between methods and built-in functions.
# Sometimes, there won't be a 1:1 correspondence.

# Changing the data type
t_m$to(dtype = "float64")
t_m$to(dtype = torch_float64())
# torch_tensor(m, dtype = "float64")


# tensor computations happen on it's current device
mps_tensor <- t_m$to(device="mps") # only available on ARM Macs
# on NVIDIA GPU's try instead : t_m$to(device="cuda") 
# This will use NVIDIA's CUDA library to run computations on the tensor

# 'cuda_is_available()' is the function that tells you if CUDA is available

out <- torch_mm(mps_tensor, mps_tensor$t())
out$to(device="cpu")

cpu_tensor <- torch_randn(10000, 10000)
mps_tensor <- cpu_tensor$to(device="mps")

bench::mark(
  mps = torch_mm(mps_tensor, mps_tensor$t()),
  cpu = torch_mm(cpu_tensor, cpu_tensor$t())
)