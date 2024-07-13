#===============================================================================
# FUNCTIONS FOR CREATING DATALOADERS
#===============================================================================

#=== MAIN FUNCTION =============================================================

#' Create a New Dataloader for Building a Scorch Model
#'
#' @description
#' This function creates a dataloader for building a scorch model from the
#' given input and output tensors.
#'
#' @param input A tensor representing the input data (usually your predictors,
#' first dimensions is index).
#'
#' @param output A tensor representing the output data (usually what you are
#' predicting, first dimension is index).
#'
#' @param name A character string representing the name of the dataloader.
#' Default is "dl".
#'
#' @param batch_size An integer specifying the batch size. Default is 32.
#'
#' @param shuffle A logical value indicating whether to shuffle the data.
#' Default is TRUE.
#'
#' @return The dataloader with the specified batch size as a scorch_dataloader.
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

scorch_create_dataloader <- function(input, output,

  name = "dl", batch_size = 32, shuffle = TRUE){

  ## Check Input Arguments

  stopifnot("`input` must be a tensor"  = ("torch_tensor" %in% class(input)))

  stopifnot("`output` must be a tensor" = ("torch_tensor" %in% class(output)))

  ## Set Up Dataset Creator Function

  create_dataset <- torch::dataset(

    name = name,

    initialize = function(input, output, aux) {

      self$input = input

      self$output = output
    },

    .getitem = function(index) {

      list(

        input = self$input[index],

        output = self$output[index])
    },

    .length = function() {

      self$input$shape[1]
    }
  )

  ## Create Dataset

  ds <- create_dataset(input, output, aux)

  ## Create the Dataloader

  dl <- torch::dataloader(ds, batch_size = batch_size, shuffle = shuffle)

  dl <- create_scorch_dataloader_class(dl)

  return(dl)
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
#' dl <- scorch_create_dataloader(input,output,batch_size=2)
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
#' dl <- scorch_create_dataloader(input,output)
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
