#===============================================================================
# FUNCTIONS FOR CREATING DATALOADERS
#===============================================================================

#=== MAIN FUNCTION =============================================================

#-------------------------------------------------------------------------------
# NOTES:
#
# Main updates were:
#
#   1. To handle multiple inputs/outputs
#   2. To make the function more flexible to the form of the data (i.e., now do
#      not need to convert to tensor first)
#
# Separately handles input vs output conversion:
#
#   1. Inputs to float, add channel dims as needed
#   2. Outputs to long (classification) or float (regression)
#   3. 1‑based slicing preserves all dims except batch
#-------------------------------------------------------------------------------

#' Create a Scorch DataLoader
#'
#' Convert R vectors or tensors into a `torch::dataloader` with automatic dtype
#' and dimension handling.
#'
#' @param input A torch tensor, R numeric vector, or named list thereof.
#'
#' @param output A torch tensor, R numeric vector, or named list thereof.
#'
#' @param batch_size Integer batch size (default 32).
#'
#' @param shuffle Logical; shuffle each epoch (default `TRUE`).
#'
#' @param num_workers Number of worker processes (default 0).
#'
#' @param pin_memory Logical; pin memory (default `FALSE`).
#'
#' @param ... Additional args passed to `torch::dataloader()`.
#'
#' @return A `torch::dataloader` with `$input` and `$output` named lists.
#'
#' @examples
#'
#' dl <- scorch_create_dataloader(
#'
#'   input  = list(a = mtcars$hp, b = mtcars$disp),
#'   output = mtcars$mpg,
#'   batch_size = 8
#' )
#'
#' batch <- iter_next(dl)
#'
#' str(batch)
#'
#' @import torch
#'
#' @export

scorch_create_dataloader <- function(

    input,
    output,
    batch_size = 32,
    shuffle = TRUE,
    num_workers = 0,
    pin_memory = FALSE,
    ...) {

    #- Wrap bare tensors/arrays into named lists

    if (!is.list(input)  || inherits(input,  "torch_tensor")) {

        input  <- list(input = input)
    }

    if (!is.list(output) || inherits(output, "torch_tensor")) {

        output <- list(output = output)
    }

    #- Convert inputs to float, add channels

    make_input_tensor <- function(x) {

        #- To torch_tensor if needed

        t <- if (inherits(x, "torch_tensor")) {

            x

        } else {

            torch::torch_tensor(x, dtype = torch_float())
        }

        #- Cast any non‑float to float

        if (t$dtype != torch_float()) {

            t <- t$to(dtype = torch_float())
        }

        #- Unsqueeze channel dim for images/features

        if (t$dim() == 3L) {

            t <- t$unsqueeze(2) # (N,H,W) to (N,1,H,W)

        } else if (t$dim() == 1L) {

            t <- t$unsqueeze(2) # (N) to (N,1)
        }

        t
    }

    #- Convert outputs to long or float appropriately

    make_output_tensor <- function(x) {

        #- If torch_tensor, keep dtype for classification/regression logic

        if (inherits(x, "torch_tensor")) {

            t <- x

        } else if (is.integer(x)) {

            t <- torch::torch_tensor(x, dtype = torch_long())

        } else {

            t <- torch::torch_tensor(x, dtype = torch_float())
        }

        #- Classification: 1‑D long stays (N)

        if (t$dtype == torch_long() && t$dim() == 1L) {

            return(t)
        }

        #- Regression single‑output: 1‑D float  to (N,1)

        if (t$dtype == torch_float() && t$dim() == 1L) {

            return(t$unsqueeze(2))
        }

        #- Regression multi‑output: multi‑dim float stays

        if (t$dtype == torch_float() && t$dim() > 1L) {

            return(t)
        }

        #- If multi‑dim long (accidental int matrix), cast to float

        if (t$dtype == torch_long() && t$dim() > 1L) {

            return(t$to(dtype = torch_float()))
        }

        t
    }

    #- Apply conversions

    input <- lapply(input, make_input_tensor)

    output <- lapply(output, make_output_tensor)

    #- Define dataset

    scorch_ds <- torch::dataset(

        name = "scorch_dataset",

        initialize = function(input, output) {

            self$input <- input

            self$output <- output

            self$n <- input[[1]]$size()[1]
        },

        .getitem = function(i) {

            slice <- function(x) {

                if (x$dim() == 1L) {

                    x[i]  # 1‑D  to scalar or 1‑D

                } else {

                    x$narrow(1, i, 1)$squeeze(1)
                }
            }

            inp <- lapply(self$input,  slice)

            out <- lapply(self$output, slice)

            list(input = inp, output = out)
        },

        .length = function() self$n
    )

    ds <- scorch_ds(input = input, output = output)

    #- Build dataloader

    torch::dataloader(

        ds,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        ...
    )
}

#=== UTILITY FUNCTIONS =========================================================

#--- SCORCH DATALOADER CLASS ---------------------------------------------------

#' Create a Scorch Dataloader Class
#'
#' @description
#' This function adds the class 'scorch_dataloader' to a given dataloader
#' object.
#'
#' @param dl A dataloader object.
#'
#' @return The input dataloader object with the class attribute set to include
#' 'scorch_dataloader'.
#'
#' @export
#'
#' @examples
#'
#' input  <- mtcars |> as.matrix() |> torch::torch_tensor()
#'
#' output <- mtcars |> as.matrix() |> torch::torch_tensor()
#'
#' dl <- scorch_create_dataloader(input, output, batch_size = 2)
#'
#' scorch_dataloader <- create_scorch_dataloader_class(dl)
#'
#' class(scorch_dataloader)

create_scorch_dataloader_class <- function(dl) {

  tmp <- dl

  class(tmp) <- c("scorch_dataloader", class(dl))

  return(tmp)
}

#--- CALCULATE TENSOR DIMENSIONS -----------------------------------------------

#' Calculate Dimensions of a Tensor
#'
#' @description
#' Calculates the dimensions of a given tensor, returning a formatted string.
#'
#' @param tensor A tensor object.
#'
#' @return A string representing the dimensions of the tensor, excluding the
#' batch dimension. If the tensor has only one dimension, returns 1.
#'
#' @export
#'
#' @examples
#'
#' input <- mtcars |> as.matrix() |> torch::torch_tensor()
#'
#' calc_dim(input)

calc_dim <- function(tensor) {

  ndim = length(dim(tensor))

  if(ndim == 1) {

    return(1)

  } else {

    return(paste0(dim(tensor)[-1], collapse = " "))
  }
}

#=== METHODS ===================================================================

#--- HEAD ----------------------------------------------------------------------

#' @importFrom utils head
#' @export
utils::head

#' Head Method for Scorch Dataloader
#'
#' @description
#' Defines the head method for objects of class 'scorch_dataloader', returning
#' the first elements of the input and output data.
#'
#' @param x An object of class 'scorch_dataloader'.
#'
#' @param ... Additional arguments to be passed to the head function.
#'
#' @return A list containing the first elements of the input and output data
#' from the dataloader.
#'
#' @export
#'
#' @examples
#'
#' input  <- mtcars |> as.matrix() |> torch::torch_tensor()
#'
#' output <- mtcars |> as.matrix() |> torch::torch_tensor()
#'
#' dl <- scorch_create_dataloader(input, output, batch_size = 2)
#'
#' head(dl)

head.scorch_dataloader <- function(x, ...) {

  val <- x$.iter()$.next()

  return(

    list(input  = head(val$input,  ...),

         output = head(val$output, ...))
  )
}

#--- PRINT ---------------------------------------------------------------------

#' Print Method for Scorch Dataloader
#'
#' @description
#' Defines the print method for objects of class 'scorch_dataloader', providing
#' a summary of its features.
#'
#' @param x An object of class 'scorch_dataloader'.
#'
#' @param ... Additional arguments to be passed to the print function.
#'
#' @return NULL. This function is called for its side effect of printing
#' information about the dataloader.
#'
#' @export
#'
#' @examples
#'
#' input  <- mtcars |> as.matrix() |> torch::torch_tensor()
#'
#' output <- mtcars |> as.matrix() |> torch::torch_tensor()
#'
#' dl <- scorch_create_dataloader(input, output)
#'
#' print(dl)

print.scorch_dataloader <- function(x, ...) {

  cat("This is a dataloader object with features:\n")

  cat(paste0(" * Batch size: ",

    crayon::red(x$batch_size)))

  cat("\n")

  cat(paste0(" * Number of batches: ",

    crayon::red(x$.length())))

  cat("\n")

  cat(paste0(" * Dimension of input tensors: ",

    crayon::red(calc_dim(x$.iter()$.next()$input))))

  cat("\n")

  cat(paste0(" * Dimension of output tensors: ",

    crayon::red(calc_dim(x$.iter()$.next()$output))))
}

#=== END =======================================================================
