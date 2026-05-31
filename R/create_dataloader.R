#===============================================================================
# FUNCTION TO CREATE A SCORCH DATALOADER
#===============================================================================

#=== MAIN FUNCTION =============================================================

#' Create a Scorch DataLoader for Multi-Input / Multi-Output Models
#'
#' @description
#' Builds a \code{torch::dataloader} from input and output data. Accepts
#' bare tensors, R arrays, or named lists for multi-input and
#' multi-output architectures. Existing torch tensor dtypes are preserved by
#' default; R arrays are converted to float unless they are integer outputs.
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
#' @param input_dtype Optional torch dtype for inputs. Defaults to
#'   \code{"auto"}, which preserves existing torch tensor dtypes and converts
#'   non-tensor inputs to \code{torch::torch_float()}. Use
#'   \code{torch::torch_long()} to force token indices for embedding layers.
#'
#' @param output_dtype Optional torch dtype for outputs. Defaults to
#'   \code{"auto"}, which preserves existing torch tensor dtypes, keeps integer
#'   outputs as \code{torch_long()}, and uses \code{torch_float()} for numeric
#'   outputs. Use \code{torch::torch_long()} to force sequence targets in
#'   language models.
#'
#' @param ... Additional arguments passed to \code{torch::dataloader()}.
#'
#' @returns A \code{torch::dataloader} yielding batches of the form
#'   \code{list(input = <named list>, output = <named list>)}.
#'
#' @details
#' Type conversion is handled automatically:
#'   \describe{
#'     \item{Inputs}{Existing torch tensor dtypes are preserved in
#'       \code{"auto"} mode. 3-D float tensors (N, H, W) gain a channel
#'       dimension (N, 1, H, W). 1-D float tensors (N) become (N, 1).}
#'     \item{Outputs}{Integer outputs and long tensors are kept as
#'       \code{torch_long()}. Float outputs are kept as \code{torch_float()}.
#'       1-D float tensors are unsqueezed to (N, 1). Explicit
#'       \code{output_dtype} overrides this behavior.}
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
                                     input_dtype  = "auto",
                                     output_dtype = "auto",
                                     ...) {

  #- Wrap bare tensors/arrays into named lists.

  if (!is.list(input) || inherits(input, "torch_tensor")) {

    input <- list(input = input)
  }

  if (!is.list(output) || inherits(output, "torch_tensor")) {

    output <- list(output = output)
  }

  cast_tensor <- function(t, dtype) {
    if (is.null(dtype)) {
      return(t)
    }

    if (is.character(dtype)) {
      if (identical(dtype, "auto")) {
        return(t)
      }
      stop("`input_dtype` and `output_dtype` must be torch dtypes, NULL, or ",
           "`output_dtype = \"auto\"`.", call. = FALSE)
    }

    if (t$dtype != dtype) {
      t <- t$to(dtype = dtype)
    }

    t
  }

  #- Convert inputs, add channel dims for float image/feature tensors.

  make_input_tensor <- function(x) {

    t <- if (inherits(x, "torch_tensor")) {

      x

    } else {

      dtype <- if (is.null(input_dtype) || identical(input_dtype, "auto")) {
        torch::torch_float()
      } else if (is.character(input_dtype)) {
        stop("`input_dtype` must be a torch dtype, NULL, or \"auto\".",
             call. = FALSE)
      } else {
        input_dtype
      }
      torch::torch_tensor(x, dtype = dtype)
    }

    t <- cast_tensor(t, input_dtype)

    #- Unsqueeze channel dim only for float image/feature tensors.

    if (t$dtype == torch::torch_float() && t$dim() == 3L) {

      t <- t$unsqueeze(2)  # (N, H, W) -> (N, 1, H, W)

    } else if (t$dtype == torch::torch_float() && t$dim() == 1L) {

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

    if (!identical(output_dtype, "auto")) {
      t <- cast_tensor(t, output_dtype)

      if (t$dtype == torch::torch_float() && t$dim() == 1L) {
        return(t$unsqueeze(2))
      }

      return(t)
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

    #- Sequence/classification targets can be multi-dimensional long tensors.

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
