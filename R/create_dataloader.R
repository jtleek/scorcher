#===============================================================================
# FUNCTION TO CREATE A SCORCH DATALOADER
#===============================================================================

#=== MAIN FUNCTION =============================================================

#' Create a Scorch DataLoader for Multi-Input / Multi-Output Models
#'
#' @description
#' Builds a \code{torch::dataloader} from input and output data. Accepts
#' bare tensors, R arrays, or named lists for multi-input and
#' multi-output architectures. Inputs are converted to float and outputs
#' are converted to the appropriate dtype (long for classification,
#' float for regression).
#'
#' @param input A torch tensor, R array, or named list thereof. For
#'   multi-input models, pass a named list with one element per input
#'   stream (e.g., \code{list(a = tensor_a, b = tensor_b)}).
#'
#' @param output A torch tensor, R array, or named list thereof. For
#'   multi-output models, pass a named list with one element per output
#'   head.
#'
#' @param batch_size Integer. Batch size (default 32).
#'
#' @param shuffle Logical. Shuffle each epoch? (default \code{TRUE}).
#'
#' @param num_workers Integer. Number of worker processes for data
#'   loading (default 0).
#'
#' @param pin_memory Logical. Pin memory for faster GPU transfer
#'   (default \code{FALSE}).
#'
#' @param ... Additional arguments passed to \code{torch::dataloader()}.
#'
#' @returns A \code{torch::dataloader} yielding batches of the form
#'   \code{list(input = <named list>, output = <named list>)}.
#'
#' @details
#' Type conversion is handled automatically:
#'   \describe{
#'     \item{Inputs}{Cast to \code{torch_float()}. 3-D tensors (N, H, W)
#'       gain a channel dimension (N, 1, H, W). 1-D tensors (N) become
#'       (N, 1).}
#'     \item{Outputs}{Integer vectors and 1-D long tensors are kept as
#'       \code{torch_long()} for classification. Float vectors and tensors
#'       are kept as \code{torch_float()} for regression. 1-D float
#'       tensors are unsqueezed to (N, 1).}
#'   }
#'
#' Bare (non-list) inputs and outputs are wrapped in a single-element
#' named list automatically.
#'
#' @examples
#' \dontrun{
#' # Single input / single output
#' dl <- scorch_create_dataloader(
#'   input  = torch::torch_randn(100, 10),
#'   output = torch::torch_randn(100, 1),
#'   batch_size = 32
#' )
#'
#' # Multi-input for fusion models
#' dl <- scorch_create_dataloader(
#'   input  = list(a = tensor_a, b = tensor_b),
#'   output = labels,
#'   batch_size = 16
#' )
#' }
#'
#' @family data loading
#'
#' @export

scorch_create_dataloader <- function(input,
                                     output,
                                     batch_size  = 32,
                                     shuffle     = TRUE,
                                     num_workers = 0,
                                     pin_memory  = FALSE,
                                     ...) {

  #- Wrap bare tensors/arrays into named lists.

  if (!is.list(input) || inherits(input, "torch_tensor")) {

    input <- list(input = input)
  }

  if (!is.list(output) || inherits(output, "torch_tensor")) {

    output <- list(output = output)
  }

  #- Convert inputs to float, add channel dims as needed.

  make_input_tensor <- function(x) {

    t <- if (inherits(x, "torch_tensor")) {

      x

    } else {

      torch::torch_tensor(x, dtype = torch::torch_float())
    }

    #- Cast any non-float to float.

    if (t$dtype != torch::torch_float()) {

      t <- t$to(dtype = torch::torch_float())
    }

    #- Unsqueeze channel dim for images/features.

    if (t$dim() == 3L) {

      t <- t$unsqueeze(2)  # (N, H, W) -> (N, 1, H, W)

    } else if (t$dim() == 1L) {

      t <- t$unsqueeze(2)  # (N) -> (N, 1)
    }

    t
  }

  #- Convert outputs to long or float appropriately.

  make_output_tensor <- function(x) {

    if (inherits(x, "torch_tensor")) {

      t <- x

    } else if (is.integer(x)) {

      t <- torch::torch_tensor(x, dtype = torch::torch_long())

    } else {

      t <- torch::torch_tensor(x, dtype = torch::torch_float())
    }

    #- Classification: 1-D long stays (N).

    if (t$dtype == torch::torch_long() && t$dim() == 1L) {

      return(t)
    }

    #- Regression single-output: 1-D float -> (N, 1).

    if (t$dtype == torch::torch_float() && t$dim() == 1L) {

      return(t$unsqueeze(2))
    }

    #- Regression multi-output: multi-dim float stays.

    if (t$dtype == torch::torch_float() && t$dim() > 1L) {

      return(t)
    }

    #- Multi-dim long (accidental int matrix): cast to float.

    if (t$dtype == torch::torch_long() && t$dim() > 1L) {

      return(t$to(dtype = torch::torch_float()))
    }

    t
  }

  #- Apply conversions.

  input  <- lapply(input, make_input_tensor)

  output <- lapply(output, make_output_tensor)

  #- Define dataset.

  scorch_ds <- torch::dataset(

    name = "scorch_dataset",

    initialize = function(input, output) {

      self$input  <- input

      self$output <- output

      self$n <- input[[1]]$size()[1]
    },

    .getitem = function(i) {

      slice <- function(x) {

        if (x$dim() == 1L) {

          x[i]

        } else {

          x$narrow(1, i, 1)$squeeze(1)
        }
      }

      inp <- lapply(self$input, slice)

      out <- lapply(self$output, slice)

      list(input = inp, output = out)
    },

    .length = function() self$n
  )

  ds <- scorch_ds(input = input, output = output)

  #- Build dataloader.

  torch::dataloader(
    ds,
    batch_size  = batch_size,
    shuffle     = shuffle,
    num_workers = num_workers,
    pin_memory  = pin_memory,
    ...
  )
}

#=== END =======================================================================
