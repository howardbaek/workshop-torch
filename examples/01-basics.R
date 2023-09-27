library(torch)

m <- matrix(runif(1000), nrow = 100)
a <- array(runif(10000), dim = c(100, 10, 10))

# create torch tensors from R objects

t_m <- torch_tensor(m, dtype = "double")
t_a <- torch_tensor(a)

# back to R

as.array(t_m)
as.matrix(t_m)

# attributes of a tensor

t_m$device
t_m$dtype
t_m$shape # dim(t_m)
t_m$requires_grad

# methods

(t_m -1)$abs()
t_m$add(1)
t_m$sum()
t_m$min()

# in-place methods end in `_`

t_m$add_(1)
t_m # t_m is modified in place, no copy is made

# built-in functions

torch_mm(t_m, t_m$t())
torch_sum(t_m)
torch_mean(t_m)

# changing the data type

t_m$to(dtype = "float64")
t_m$to(dtype = torch_float64())
# torch_tensor(m, dtype = "float64")

# tensor computations happen on it's current device

mps_tensor <- t_m$to(device="mps") # only available on ARM Macs
# on GPU's try instead : t_m$to(device="cuda") 

out <- torch_mm(mps_tensor, mps_tensor$t())
out$to(device="cpu")

cpu_tensor <- torch_randn(10000, 10000)
mps_tensor <- cpu_tensor$to(device="mps")

bench::mark(
  mps = torch_mm(mps_tensor, mps_tensor$t()),
  cpu = torch_mm(cpu_tensor, cpu_tensor$t())
)

