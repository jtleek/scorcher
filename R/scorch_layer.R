#===============================================================================
# SCORCH LAYER
#===============================================================================

#=== MAIN FUNCTION =============================================================

#' Add a Layer to a Scorch Model
#'
#' @description
#' This function adds a neural network layer or a residual block to a scorch
#' model architecture. The layer can include a single activation or a series
#' of layers specified by a vector of layer types.
#'
#' @param scorch_model A scorch model object to which the layer or block will
#' be added.
#'
#' @param layer_type A string or character vector specifying the type of
#' layer to add (e.g., \code{c("conv2d", "gelu", "linear", "relu")}).
#' Elements will be converted to the corresponding torch layer function
#' (e.g., \code{nn_linear}, \code{nn_conv2d}).
#'
#' @param in_features Optional. An integer specifying the number of input
#' features for the layer. Used for layers such as \code{nn_linear}. Default
#' is \code{NULL}.
#'
#' @param out_features Optional. An integer specifying the number of output
#' features for the layer. Used for layers such as \code{nn_linear}. Default
#' is \code{NULL}.
#'
#' @param use_residual Logical value indicating whether to use a residual
#' connection. If \code{TRUE}, the function adds a residual block of the
#' form \code{x + f(g(x))}. Default is \code{FALSE}.
#'
#' @param ... Additional arguments passed to the layer constructors. These can
#' include other required or optional parameters depending on the layer type.
#'
#' @details
#' The \code{layer_type} argument can accept either a single string or a vector
#' of strings. If a single string is provided, the function adds a single layer
#' of the specified type to the model. For example, if
#' \code{layer_type = "conv2d"}, the function will add a 2D convolutional layer.
#'
#' If a vector of strings is provided, the function creates a sequential block
#' consisting of multiple layers, where each element in the vector specifies a
#' layer or activation function to include in the block. For example, if
#' \code{layer_type = c("conv2d", "gelu", "linear", "relu")}, the function will
#' add a block that first applies a 2D convolution, then the GELU activation,
#' followed by a linear layer, and finally the ReLU activation.
#'
#' When \code{use_residual = TRUE}, the function constructs a residual
#' connection block of the form \code{x + f(g(x))}, where \code{f} and
#' \code{g} represent the layers and activations specified by \code{layer_type}.
#'
#' A residual connection is a technique where the input to a block of layers
#' is added directly to the block's output. This creates a shortcut path, or
#' "skip connection," that allows the original input to bypass one or more
#' layers. Residual connections are beneficial for training deep networks
#' because they help mitigate the vanishing gradient problem by allowing
#' gradients to flow more easily through the network.
#'
#' @return Returns the updated scorch model with the new layer or block
#' added to its architecture.
#'
#' @examples
#'
#' input  <- mtcars |> as.matrix() |> torch::torch_tensor()
#'
#' output <- mtcars |> as.matrix() |> torch::torch_tensor()
#'
#' dl <- scorch_create_dataloader(input, output, batch_size = 2)
#'
#' scorch_model <- dl |> initiate_scorch() |>
#'
#'   scorch_layer(layer_type = "conv2d", in_features = 16, out_features = 32)
#'
#' scorch_model <- scorch_model |>
#'
#'   scorch_layer(layer_type = c("conv2d", "gelu", "linear", "relu"),
#'
#'    in_features = 16, out_features = 32, use_residual = TRUE)
#'
#' @export

scorch_layer <- function(scorch_model, layer_type,

  in_features = NULL, out_features = NULL, use_residual = FALSE, ...) {

  # Convert layer_type to lowercase to ensure consistency

  layer_type <- tolower(layer_type)

  # Check if all layer types are valid

  valid_layers <- sapply(layer_type, function(layer) {

    function_name <- paste0("nn_", layer)

    exists(function_name, envir = asNamespace("torch"))
  })

  if (!all(valid_layers)) {

    invalid_layers <- layer_type[!valid_layers]

    stop(paste("Invalid layer types detected:",

      paste(invalid_layers, collapse = ", ")))
  }

  # Create layers

  layers <- lapply(layer_type, function(layer) {

    function_name <- paste0("nn_", layer)

    nn_function <- get(function_name, envir = asNamespace("torch"))

    # Collect optional arguments

    args_list <- list(...)

    if ("in_features" %in% names(formals(nn_function))) {

      args_list$in_features <- in_features
    }

    if ("out_features" %in% names(formals(nn_function))) {

      args_list$out_features <- out_features
    }

    do.call(nn_function, args_list)
  })

  if (use_residual) {

    # If residual connection is used, wrap the layers in a residual block

    residual_block <- nn_module(

      initialize = function() {

        self$block <- nn_sequential(!!!layers)
      },

      forward = function(x) {

        x_residual <- self$block(x)

        x + x_residual

        # x + self$block(x)
      }
    )

    res_layer <- residual_block()

    class(res_layer) <- c("residual_block", class(res_layer))

    scorch_model$scorch_architecture <- append(

      scorch_model$scorch_architecture, list(res_layer, type = "layer"))

  } else {

    # Add the sequential block of layers to the model

    for (i in seq_along(layers)) {

      scorch_model$scorch_architecture <- append(

        scorch_model$scorch_architecture, list(layers[[i]], type = "layer"))
    }
  }

  scorch_model
}

#=== END =======================================================================
