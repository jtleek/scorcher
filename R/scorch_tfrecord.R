#===============================================================================
# FUNCTION FOR LOADING TFRecord DATA FILES
#===============================================================================

#======= MAIN FUNCTION =========================================================

#' Create a torch dataset from TFRecord files
#'
#' This function reads embedding data from TFRecord files and returns torch
#' tensors in a format compatible with scorcher pipelines, similar to
#' torchvision datasets.
#'
#' @param filepaths A character vector of file paths to .tfrecord files
#'
#' @param input A character string specifying the feature name in the TFRecord.
#' Default is "embedding"
#'
#' @param output A numeric vector of outcome labels (0-based or 1-based)
#'
#' @param dtype A character string specifying the data type for parsing
#' features. Default is "float32"
#'
#' @param verbose A logical value indicating whether to print progress messages.
#'  Default is TRUE
#'
#' @return A list of class "scorch_tfrecord" containing:
#'
#' \itemize{
#'
#'   \item \code{input}: A torch tensor of embeddings with shape
#'   (n_samples x n_features)
#'
#'   \item \code{output}: A torch tensor of labels (converted to 1-based
#'   indexing if needed)
#'
#'   \item \code{n_samples}: Number of samples loaded
#'
#'   \item \code{n_features}: Number of features per sample
#' }
#'
#' @details
#'
#' The function automatically handles:
#' \itemize{
#'
#'   \item Parsing of TFRecord files with VarLenFeature format
#'
#'   \item Conversion of 0-based labels to 1-based for torch compatibility
#'
#'   \item Validation of file paths and removal of missing files
#' }
#'
#' The default feature name is "embedding", but this should be changed based on
#' the user's TFRecord structure. User should check the data documentation to
#' identify the correct feature name. Common alternatives include "features",
#' "vector", or modality-specific names.
#'
#' The \code{dtype} parameter specifies the data type of features in TFRecord:
#'
#' \itemize{
#'   \item \code{"float32"} (default): 32-bit floating point, standard for
#'   neural network embeddings
#'
#'   \item \code{"float16"}: 16-bit floating point, uses half the memory but
#'   less precise
#'
#'   \item \code{"float64"}: 64-bit floating point, double precision for
#'   scientific computing
#'
#'   \item \code{"int32"}: 32-bit integer, for categorical features or counts
#'
#'   \item \code{"int64"}: 64-bit integer, for large integer values
#'
#'   \item \code{"string"}: For text data.
#' }
#'
#' The returned object is compatible with \code{scorch_create_dataloader()} and
#' can be used directly:
#'
#' \code{dataset <- scorch_tfrecord(...); dl <- scorch_create_dataloader(
#' dataset$input, dataset$output)}
#'
#' @section Required packages:
#'
#' This function requires:
#'
#' \itemize{
#'   \item \code{tensorflow} for TFRecord parsing
#'
#'   \item \code{tfdatasets} for dataset operations
#'
#'   \item \code{torch} for tensor creation
#' }
#'
#' @examples
#' \dontrun{
#' # Load metadata
#' metadata <- read.csv("path/to/metadata.csv")
#'
#' # Create dataset with default settings (float32)
#'
#' dataset <- scorch_tfrecord(metadata$filepath, output = metadata$label)
#'
#' # Create dataset silently
#'
#' dataset <- scorch_tfrecord(metadata$filepath, output = metadata$label, verbose = FALSE)
#'
#' # Access data for use with scorch_create_dataloader
#'
#' dl <- scorch_create_dataloader(dataset$input, dataset$output, batch_size = 32)
#'
#' # Create dataset with custom feature name
#'
#' custom_dataset <- scorch_tfrecord(
#'
#'   filepaths = file_list$path,
#'
#'   input = "feature_vector",
#'
#'   output = file_list$outcome
#' )
#' }
#'
# @importFrom tensorflow tf
#'
# @importFrom tfdatasets tfrecord_dataset dataset_map dataset_batch dataset_take as_array_iterator
#'
#' @export
#'
scorch_tfrecord <- function(filepaths, input = "embedding", output,

                            dtype = "float32", verbose = TRUE) {

  # ===== Dependency validation =====

  if (!requireNamespace("tensorflow", quietly = TRUE)) {

    stop("Package 'tensorflow' is required for scorch_tfrecord().

         Please install it with: install.packages('tensorflow')")
  }

  if (!requireNamespace("tfdatasets", quietly = TRUE)) {

    stop("Package 'tfdatasets' is required for scorch_tfrecord().

         Please install it with: install.packages('tfdatasets')")
  }

  # ===== Input validation =====

  # Check for empty inputs

  if (length(filepaths) == 0) {

    stop(crayon::red("filepaths cannot be empty"))
  }

  if (length(output) == 0) {

    stop(crayon::red("output cannot be empty"))
  }

  # Map string dtype to tensorflow dtype

  dtype_map <- list(

    "float16" = tensorflow::tf$float16,

    "float32" = tensorflow::tf$float32,

    "float64" = tensorflow::tf$float64,

    "int8" = tensorflow::tf$int8,

    "int16" = tensorflow::tf$int16,

    "int32" = tensorflow::tf$int32,

    "int64" = tensorflow::tf$int64,

    "uint8" = tensorflow::tf$uint8,

    "uint16" = tensorflow::tf$uint16,

    "bool" = tensorflow::tf$bool,

    "string" = tensorflow::tf$string
  )

  # Convert string dtype to tensorflow dtype
  # Check if dtype is a string and needs conversion

  if (is.character(dtype)) {

    # Validate dtype string against allowed options

    if (!dtype %in% names(dtype_map)) {

      stop(crayon::red("Invalid dtype '", dtype, "'. Valid options are: ",

                       paste(names(dtype_map), collapse = ", ")))
    }

    tf_dtype <- dtype_map[[dtype]]

  } else {

    # Allow direct tensorflow dtype objects for backward compatibility

    tf_dtype <- dtype
  }

  # Validate inputs
  # Check filepaths is character vector

  if (!is.character(filepaths)) {

    stop(crayon::red("filepaths must be a character vector"))
  }

  # Check input is single string

  if (!is.character(input) || length(input) != 1) {

    stop(crayon::red("input must be a single character string"))
  }

  # Check output is numeric

  if (!is.numeric(output) && !is.integer(output)) {

    stop(crayon::red("output must be numeric or integer vector"))
  }

  # Check for invalid output values (NA, NaN, Inf)

  if (any(!is.finite(output))) {

    stop(crayon::red("output contains invalid values (NA, NaN, or Inf)"))
  }

  # Check matching lengths

  if (length(filepaths) != length(output)) {

    stop(crayon::red("Length of filepaths (", length(filepaths),

                     ") must match length of output (", length(output), ")"))
  }

  # ===== File validation =====

  # Check for missing files

  missing_files <- !file.exists(filepaths)

  # Remove missing files if any found

  if (any(missing_files)) {

    if (verbose) {

      warning("Found ", sum(missing_files), " missing files. Removing them from

              dataset.")
    }

    filepaths <- filepaths[!missing_files]

    output <- output[!missing_files]
  }

  # Ensure at least one valid file remains

  if (length(filepaths) == 0) {

    stop(crayon::red("No valid filepaths found!"))
  }

  # ===== TFRecord parsing =====

  # Define TFRecord parser
  parse_embedding_fn <- function(proto) {

    features <- list()

    features[[input]] <- tensorflow::tf$io$VarLenFeature(dtype = tf_dtype)

    parsed_features <- tensorflow::tf$io$parse_single_example(proto, features)

    embedding_vector <- tensorflow::tf$sparse$to_dense(parsed_features[[input]])

    tensorflow::tf$reshape(embedding_vector, shape = list(-1L))
  }

  # Try-catch wrapper for TensorFlow operations

  all_embeddings <- tryCatch({

    # Create dataset

    tf_dataset <- tfdatasets::tfrecord_dataset(filepaths) |>

      tfdatasets::dataset_map(parse_embedding_fn) |>

      tfdatasets::dataset_batch(length(filepaths))  # Batch all at once

    # Get the single batch containing all data

    all_data <- tf_dataset |>

      tfdatasets::dataset_take(1) |>

      tfdatasets::as_array_iterator() |>

      (\(x) x$`__next__`())()

    # Convert to matrix

    as.matrix(all_data)

  }, error = function(e) {

    stop(crayon::red("Error reading TFRecord files: ", e$message))
  })

  #--- Check if the features were loaded properly

  # Check for zero features and issue a warning

  if (ncol(all_embeddings) == 0) {

    warning(crayon::yellow("Warning: Loaded ", nrow(all_embeddings),

                           " samples with 0 features.\n",

                           "Please check that the `input` argument ('", input,

                           "') is the correct feature name in your TFRecord files."))
  }

  # ===== Label processing =====

  # Fix label indexing for torch

  # Convert 0-based to 1-based if minimum label is 0

  if (min(output) == 0) {

    if (verbose) {

      cat("Converting 0-based labels to 1-based indexing for torch

          compatibility\n")

      cat("Label conversion: 0 -> 1, 1 -> 2, ...\n")
    }

    output <- output + 1
  }

  # ===== Tensor creation =====

  # Create torch tensors with appropriate dtype
  # Check if dtype is floating point

  if (dtype %in% c("float16", "float32", "float64")) {

    input_tensor <- torch::torch_tensor(all_embeddings, dtype = torch::torch_float())

    # Check if dtype is integer

  } else if (dtype %in% c("int8", "int16", "int32", "int64")) {

    input_tensor <- torch::torch_tensor(all_embeddings, dtype = torch::torch_long())

  } else {

    # Default to float for other types (e.g., string, bool)

    input_tensor <- torch::torch_tensor(all_embeddings, dtype = torch::torch_float())
  }

  output_tensor <- torch::torch_tensor(output, dtype = torch::torch_long())

  # Print loading summary if verbose = TRUE

  if (verbose) {

    cat("Loaded", nrow(all_embeddings), "embeddings with", ncol(all_embeddings),

        "features each from feature '", input, "'\n")
  }

  # ===== Create output object =====

  # Create object

  result <- list(

    input = input_tensor,

    output = output_tensor,

    n_samples = nrow(all_embeddings),

    n_features = ncol(all_embeddings)
  )

  # Add class using utility function

  result <- create_scorch_tfrecord_class(result)

  return(result)
}

#=== UTILITY FUNCTIONS =========================================================

#--- SCORCH TFRECORD CLASS -----------------------------------------------------

#' Create a Scorch TFRecord Class
#'
#' @description
#' This function adds the class 'scorch_tfrecord' to a given tfrecord object.
#'
#' @param tfrecord A tfrecord object.
#'
#' @return The input tfrecord object with the class attribute set to include
#' 'scorch_tfrecord'.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Create basic tfrecord object
#'
#' tfrecord <- list(
#'
#'   input = torch::torch_tensor(matrix(1:12, nrow = 3)),
#'
#'   output = torch::torch_tensor(c(1, 2, 1)),
#'
#'   n_samples = 3,
#'
#'   n_features = 4
#' )
#'
#' scorch_tfrecord <- create_scorch_tfrecord_class(tfrecord)
#'
#' class(scorch_tfrecord)
#' }
#'
create_scorch_tfrecord_class <- function(tfrecord) {

  tmp <- tfrecord

  class(tmp) <- c("scorch_tfrecord", class(tfrecord))

  return(tmp)
}

#=== METHODS ===================================================================

#--- HEAD ----------------------------------------------------------------------

#' @importFrom utils head
#' @export
utils::head

#' Head Method for Scorch TFRecord
#'
#' @description
#' Defines the head method for objects of class 'scorch_tfrecord', returning
#' the first elements of the input and output data.
#'
#' @param x An object of class 'scorch_tfrecord'.
#'
#' @param ... Additional arguments to be passed to the head function.
#'
#' @return A list containing the first elements of the input and output data
#' from the tfrecord dataset.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' dataset <- scorch_tfrecord(filepaths, output = labels)
#'
#' head(dataset)
#' }
head.scorch_tfrecord <- function(x, ...) {

  return(

    list(input  = head(x$input,  ...),

         output = head(x$output, ...))
  )
}

#--- PRINT ---------------------------------------------------------------------

#' Print Method for Scorch TFRecord
#'
#' @description
#' Defines the print method for objects of class 'scorch_tfrecord', providing
#' a summary of its features.
#'
#' @param x An object of class 'scorch_tfrecord'.
#'
#' @param ... Additional arguments to be passed to the print function.
#'
#' @return NULL. This function is called for its side effect of printing
#' information about the tfrecord dataset.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' dataset <- scorch_tfrecord(metadata$filepaths, output = metadata$labels)
#'
#' print(dataset)
#' }
#'
print.scorch_tfrecord <- function(x, ...) {

  cat("This is a scorch_tfrecord dataset with features:\n")

  cat(paste0(" * Number of samples: ", crayon::red(x$n_samples)))

  cat("\n")

  cat(paste0(" * Number of features: ", crayon::red(x$n_features)))

  cat("\n")

  cat(paste0(" * Dimension of input tensors: ",

             crayon::red(paste(dim(x$input), collapse = " x "))))

  cat("\n")

  cat(paste0(" * Dimension of output tensors: ",

             crayon::red(paste(dim(x$output), collapse = " x "))))
}

#=== END =======================================================================
