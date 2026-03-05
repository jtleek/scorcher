#===============================================================================
# FUNCTION FOR LOADING TFRecord DATA FILES
#===============================================================================

#=== MAIN FUNCTION =============================================================

#' Create a Torch Dataset from TFRecord Files
#'
#' @description
#' Reads embedding data from TFRecord files and returns torch tensors in a
#' format compatible with scorcher pipelines, similar to torchvision datasets.
#'
#' @param filepaths A character vector of file paths to \code{.tfrecord} files.
#'
#' @param input A character string specifying the feature name in the TFRecord.
#'   Default is \code{"embedding"}.
#'
#' @param output A numeric vector of outcome labels (0-based or 1-based).
#'
#' @param dtype A character string specifying the data type for parsing
#'   features. One of \code{"float32"} (default), \code{"float16"},
#'   \code{"float64"}, \code{"int32"}, \code{"int64"}, \code{"int8"},
#'   \code{"int16"}, \code{"uint8"}, \code{"uint16"}, \code{"bool"}, or
#'   \code{"string"}.
#'
#' @param verbose Logical. If \code{TRUE} (default), prints progress messages
#'   via \code{message()}.
#'
#' @returns A list of class \code{"scorch_tfrecord"} containing:
#' \itemize{
#'   \item \code{input}: A torch tensor of embeddings with shape
#'     (n_samples x n_features).
#'   \item \code{output}: A torch tensor of labels (converted to 1-based
#'     indexing if needed).
#'   \item \code{n_samples}: Number of samples loaded.
#'   \item \code{n_features}: Number of features per sample.
#' }
#'
#' @details
#' The function automatically handles:
#' \itemize{
#'   \item Parsing of TFRecord files with VarLenFeature format.
#'   \item Conversion of 0-based labels to 1-based for torch compatibility.
#'   \item Validation of file paths and removal of missing files.
#' }
#'
#' The default feature name is \code{"embedding"}, but this should be changed
#' based on the user's TFRecord structure. Check the data documentation to
#' identify the correct feature name. Common alternatives include
#' \code{"features"}, \code{"vector"}, or modality-specific names.
#'
#' The returned object is compatible with \code{\link{scorch_create_dataloader}}
#' and can be used directly:
#' \preformatted{
#' dataset <- scorch_tfrecord(...)
#' dl <- scorch_create_dataloader(dataset$input, dataset$output)
#' }
#'
#' @section Required packages:
#' This function requires the \pkg{tensorflow} and \pkg{tfdatasets} packages,
#' which are listed in \code{Suggests}. Install them with:
#' \preformatted{
#' install.packages(c("tensorflow", "tfdatasets"))
#' }
#'
#' @examples
#' \dontrun{
#' # Load metadata
#' metadata <- read.csv("path/to/metadata.csv")
#'
#' # Create dataset with default settings (float32)
#' dataset <- scorch_tfrecord(metadata$filepath, output = metadata$label)
#'
#' # Create dataset silently
#' dataset <- scorch_tfrecord(metadata$filepath, output = metadata$label,
#'                            verbose = FALSE)
#'
#' # Access data for use with scorch_create_dataloader
#' dl <- scorch_create_dataloader(dataset$input, dataset$output,
#'                                batch_size = 32)
#'
#' # Create dataset with custom feature name
#' custom_dataset <- scorch_tfrecord(
#'   filepaths = file_list$path,
#'   input = "feature_vector",
#'   output = file_list$outcome
#' )
#' }
#'
#' @family data loading
#'
#' @export

scorch_tfrecord <- function(filepaths, input = "embedding", output,
                            dtype = "float32", verbose = TRUE) {

  # ===== Dependency validation ================================================

  if (!requireNamespace("tensorflow", quietly = TRUE)) {

    stop("Package 'tensorflow' is required for scorch_tfrecord(). ",
         "Please install it with: install.packages('tensorflow')",
         call. = FALSE)
  }

  if (!requireNamespace("tfdatasets", quietly = TRUE)) {

    stop("Package 'tfdatasets' is required for scorch_tfrecord(). ",
         "Please install it with: install.packages('tfdatasets')",
         call. = FALSE)
  }

  # ===== Input validation =====================================================

  if (!is.character(filepaths)) {

    stop("`filepaths` must be a character vector.", call. = FALSE)
  }

  if (length(filepaths) == 0) {

    stop("`filepaths` cannot be empty.", call. = FALSE)
  }

  if (!is.character(input) || length(input) != 1) {

    stop("`input` must be a single character string.", call. = FALSE)
  }

  if (!is.numeric(output) && !is.integer(output)) {

    stop("`output` must be a numeric or integer vector.", call. = FALSE)
  }

  if (length(output) == 0) {

    stop("`output` cannot be empty.", call. = FALSE)
  }

  if (any(!is.finite(output))) {

    stop("`output` contains invalid values (NA, NaN, or Inf).", call. = FALSE)
  }

  if (length(filepaths) != length(output)) {

    stop("Length of `filepaths` (", length(filepaths),
         ") must match length of `output` (", length(output), ").",
         call. = FALSE)
  }

  # ===== Dtype mapping ========================================================

  dtype_map <- list(
    "float16" = tensorflow::tf$float16,
    "float32" = tensorflow::tf$float32,
    "float64" = tensorflow::tf$float64,
    "int8"    = tensorflow::tf$int8,
    "int16"   = tensorflow::tf$int16,
    "int32"   = tensorflow::tf$int32,
    "int64"   = tensorflow::tf$int64,
    "uint8"   = tensorflow::tf$uint8,
    "uint16"  = tensorflow::tf$uint16,
    "bool"    = tensorflow::tf$bool,
    "string"  = tensorflow::tf$string
  )

  if (is.character(dtype)) {

    if (!dtype %in% names(dtype_map)) {

      stop("Invalid dtype '", dtype, "'. Valid options are: ",
           paste(names(dtype_map), collapse = ", "), ".",
           call. = FALSE)
    }

    tf_dtype <- dtype_map[[dtype]]

  } else {

    # Allow direct tensorflow dtype objects for backward compatibility

    tf_dtype <- dtype
  }

  # ===== File validation ======================================================

  missing_files <- !file.exists(filepaths)

  if (any(missing_files)) {

    warning("Found ", sum(missing_files),
            " missing file(s). Removing them from dataset.",
            call. = FALSE)

    filepaths <- filepaths[!missing_files]

    output <- output[!missing_files]
  }

  if (length(filepaths) == 0) {

    stop("No valid filepaths found after removing missing files.",
         call. = FALSE)
  }

  # ===== TFRecord parsing =====================================================

  #- Define the parser function for a single TFRecord example.
  #- Each .tfrecord file contains a serialized protobuf with features stored as

  #- VarLenFeature (variable-length), which is the standard format for embeddings.
  #- The parser: (1) declares the feature schema, (2) parses the protobuf into a
  #- sparse tensor, (3) converts sparse to dense, (4) reshapes to a flat 1D vector.

  parse_embedding_fn <- function(proto) {

    features <- list()

    features[[input]] <- tensorflow::tf$io$VarLenFeature(dtype = tf_dtype)

    parsed_features <- tensorflow::tf$io$parse_single_example(proto, features)

    embedding_vector <- tensorflow::tf$sparse$to_dense(parsed_features[[input]])

    tensorflow::tf$reshape(embedding_vector, shape = list(-1L))
  }

  #- Read all TFRecord files into a single R matrix.
  #- Wrapped in tryCatch because TensorFlow operations can fail due to corrupted
  #- files, wrong feature names, or TF installation issues. We catch and re-throw
  #- a clean error message rather than an obscure TF traceback.

  all_embeddings <- tryCatch({

    #- Step 1: Build a lazy TensorFlow dataset pipeline.
    #- tfrecord_dataset() opens all files; each element is one raw serialized
    #- protobuf record (one file = one sample).
    #- dataset_map() applies our parser to each record, producing a 1D tensor
    #- of embedding values per sample (e.g., length 1376 for image embeddings).
    #- dataset_batch() collects ALL samples into one batch, producing a single
    #- tensor of shape (n_samples, n_features).

    tf_dataset <- tfdatasets::tfrecord_dataset(filepaths) |>
      tfdatasets::dataset_map(parse_embedding_fn) |>
      tfdatasets::dataset_batch(length(filepaths))

    #- Step 2: Extract the single batch into R.
    #- dataset_take(1) grabs our one batch (since we batched everything).
    #- as_array_iterator() converts TF dataset into a Python-style iterator
    #- that yields R arrays instead of TF tensors.
    #- __next__() calls Python's iterator protocol to retrieve the batch.

    all_data <- tf_dataset |>
      tfdatasets::dataset_take(1) |>
      tfdatasets::as_array_iterator() |>
      (\(x) x$`__next__`())()

    #- Step 3: Convert to a proper R matrix (n_samples x n_features)
    #- for downstream conversion to a torch tensor.

    as.matrix(all_data)

  }, error = function(e) {

    stop("Error reading TFRecord files: ", e$message, call. = FALSE)
  })

  # Check for zero features

  if (ncol(all_embeddings) == 0) {

    warning("Loaded ", nrow(all_embeddings), " samples with 0 features. ",
            "Please check that the `input` argument ('", input,
            "') is the correct feature name in your TFRecord files.",
            call. = FALSE)
  }

  # ===== Label processing =====================================================

  if (min(output) == 0) {

    if (verbose) {

      message("Converting 0-based labels to 1-based indexing for torch ",
              "compatibility (0 -> 1, 1 -> 2, ...).")
    }

    output <- output + 1
  }

  # ===== Tensor creation ======================================================

  if (dtype %in% c("float16", "float32", "float64")) {

    input_tensor <- torch::torch_tensor(all_embeddings,
                                        dtype = torch::torch_float())

  } else if (dtype %in% c("int8", "int16", "int32", "int64")) {

    input_tensor <- torch::torch_tensor(all_embeddings,
                                        dtype = torch::torch_long())

  } else {

    input_tensor <- torch::torch_tensor(all_embeddings,
                                        dtype = torch::torch_float())
  }

  output_tensor <- torch::torch_tensor(output, dtype = torch::torch_long())

  if (verbose) {

    message("Loaded ", nrow(all_embeddings), " embeddings with ",
            ncol(all_embeddings), " features each from feature '",
            input, "'.")
  }

  # ===== Create output object =================================================

  result <- list(
    input      = input_tensor,
    output     = output_tensor,
    n_samples  = nrow(all_embeddings),
    n_features = ncol(all_embeddings)
  )

  create_scorch_tfrecord_class(result)
}

#=== UTILITY FUNCTIONS =========================================================

#--- SCORCH TFRECORD CLASS -----------------------------------------------------

#' Create a scorch TFRecord Class
#'
#' @description
#' Adds the class \code{"scorch_tfrecord"} to a given list object.
#'
#' @param obj A list containing tfrecord dataset components.
#'
#' @returns The input object with class attribute set to include
#'   \code{"scorch_tfrecord"}.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' tfrecord <- list(
#'   input = torch::torch_tensor(matrix(1:12, nrow = 3)),
#'   output = torch::torch_tensor(c(1, 2, 1)),
#'   n_samples = 3,
#'   n_features = 4
#' )
#'
#' scorch_tfrecord <- create_scorch_tfrecord_class(tfrecord)
#' class(scorch_tfrecord)
#' }

create_scorch_tfrecord_class <- function(obj) {

  class(obj) <- c("scorch_tfrecord", class(obj))

  obj
}

#=== METHODS ===================================================================

#--- HEAD ----------------------------------------------------------------------

#' Head Method for scorch TFRecord
#'
#' @description
#' Returns the first elements of the input and output data from a
#' \code{scorch_tfrecord} object.
#'
#' @param x An object of class \code{"scorch_tfrecord"}.
#'
#' @param ... Additional arguments passed to \code{\link[utils]{head}}.
#'
#' @returns A list containing the first elements of the input and output tensors.
#'
#' @importFrom utils head
#' @export
#'
#' @examples
#' \dontrun{
#' dataset <- scorch_tfrecord(filepaths, output = labels)
#' head(dataset)
#' }

head.scorch_tfrecord <- function(x, ...) {

  list(input  = utils::head(x$input, ...),
       output = utils::head(x$output, ...))
}

#--- PRINT ---------------------------------------------------------------------

#' Print Method for scorch TFRecord
#'
#' @description
#' Prints a summary of a \code{scorch_tfrecord} object.
#'
#' @param x An object of class \code{"scorch_tfrecord"}.
#'
#' @param ... Additional arguments (currently unused).
#'
#' @returns The input object, invisibly.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' dataset <- scorch_tfrecord(metadata$filepaths, output = metadata$labels)
#' print(dataset)
#' }

print.scorch_tfrecord <- function(x, ...) {

  #- Helper for optional color formatting

  highlight <- if (requireNamespace("crayon", quietly = TRUE)) {

    crayon::red

  } else {

    identity
  }

  cat("This is a scorch_tfrecord dataset with features:\n")

  cat(paste0(" * Number of samples: ",
             highlight(x$n_samples), "\n"))

  cat(paste0(" * Number of features: ",
             highlight(x$n_features), "\n"))

  cat(paste0(" * Dimension of input tensors: ",
             highlight(paste(dim(x$input), collapse = " x ")), "\n"))

  cat(paste0(" * Dimension of output tensors: ",
             highlight(paste(dim(x$output), collapse = " x ")), "\n"))

  invisible(x)
}

#=== END =======================================================================
