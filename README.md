
<!-- README.md is generated from README.Rmd. Please edit that file -->

# scorcher: Blazing a trail for effortless model building with torch in R

<!-- badges: start -->
<!-- badges: end -->

### <img src="man/figures/scorcher.png" align="right" height="200" style="float:right; height:200px;"/>

### Overview

The `scorcher` package provides high-level functionality for building
and fitting deep learning models using the Torch library in R. By
simplifying model development through easy-to-use functions and ensuring
compatibility with existing R workflows and data structures, scorcher
facilitates the use of deep learning without the need for extensive
programming knowledge.

The package allows users to create, modify, and visualize models through
functions such as `initiate_scorch`, `scorch_layer`, `compile_scorch`,
and `fit_scorch`. The package flexibly handles tasks such as prediction,
classification, computer vision, diffusion, and more.

### Installation

You can install the development version of scorcher from
[GitHub](https://github.com/) with:

``` r
# install.packages("pak")
pak::pak("jtleek/scorcher")
```

### Getting Started

``` r
library(scorcher)
```

#### Example: Classifying MNIST Images

Here is an example where we build a convolutional neural network using
the
[MNIST](https://github.com/mlverse/torchvision/blob/main/R/dataset-mnist.R)
dataset, which contains 70,000 grayscale images of handwritten digits,
from 0 to 9. The goal is to classify these images into one of the 10
digit categories (0-9).

**1. Prepare the Training Data:**

``` r
library(torch)
library(torchvision)
```

``` r
#- Training Data

train_data <- mnist_dataset(
  root = tempdir(),
  download = TRUE,
  transform = transform_to_tensor)

x_train <- torch_tensor(train_data$data, dtype = torch_float()) |> 
  torch_unsqueeze(2)

y_train <- torch_tensor(train_data$targets, dtype = torch_long())
```

**Example Training Images:**

<img src="man/figures/README-plt1-1.png" width="100%" />

#### Defining the Neural Network

Next, we’ll define our neural network using the `scorcher` package.

**2. Create the Dataloader:**

``` r
#- Create the Dataloader

dl <- scorch_create_dataloader(x_train, y_train, batch_size = 500)
```

**3. Define the Scorcher Model:**

``` r
#- Define the Neural Network

scorch_model <- dl |> 
  initiate_scorch() |> 
  scorch_layer("conv2d", in_channels =  1, out_channels = 32, kernel_size = 3) |> 
  scorch_layer("relu") |>
  scorch_layer("conv2d", in_channels = 32, out_channels = 64, kernel_size = 3) |> 
  scorch_layer("relu") |>
  scorch_layer("max_pool2d", kernel_size = 2) |> 
  scorch_layer("dropout2d", p = 0.25) |>
  scorch_function(torch_flatten, start_dim = 2) |> 
  scorch_layer("linear", 9216, 128) |>
  scorch_layer("relu") |> 
  scorch_layer("linear", 128, 10)
```

**4. Compile the Model:**

``` r
#- Compile the Neural Network

compiled_scorch_model <- compile_scorch(scorch_model)
```
