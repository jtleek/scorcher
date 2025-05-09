#' Call a HuggingFace Model via Inference API and Save Config
#'
#' @param model_id HF repo ID string (e.g. "username/modelname").
#'
#' @param input A named list representing the inference payload (e.g. list(inputs="text")).
#'
#' @param api_key API token for HF (default from Sys.getenv("HF_API_KEY")).
#'
#' @param config_path File path to save downloaded config.json.
#'
#' @return A list with elements:
#'   - inference: parsed inference API response
#'   - config: parsed config.json or NULL
#'   - config_path: path where config.json was written
#'
#' @examples
#'
#' \dontrun{
#' out <- scorch_hf_call(
#'
#'   model_id = "distilbert-base-uncased",
#'   input = list(inputs = "Hello!")
#' )
#'
#' print(out$config)
#' }
#'
#' @import httr
#'
#' @import jsonlite
#'
#' @export

scorch_hf_call <- function(

    model_id,
    input,
    api_key = Sys.getenv("HF_API_KEY", unset = NA),
    config_path = file.path(tempdir(),
    paste0(gsub("[^A-Za-z0-9_]",
    "_", model_id),
    "_config.json"))) {

  if (is.na(api_key) || nzchar(api_key) == FALSE) {

    stop("Please set your HuggingFace API key via `api_key=` or ",
         "`Sys.setenv(HF_API_KEY = <token>)`.", call. = FALSE)
  }

  #- Inference call

  infer_url <- paste0("https://api-inference.huggingface.co/models/", model_id)

  infer_res <- httr::POST(

    infer_url,
    httr::add_headers(Authorization = paste("Bearer", api_key)),
    body   = input,
    encode = "json"
  )

  if (httr::status_code(infer_res) != 200) {

    stop(
      "Inference API request failed [",
      httr::status_code(infer_res), "]: ",
      httr::content(infer_res, as = "text", encoding = "UTF-8"),
      call. = FALSE
    )
  }

  inference <- httr::content(infer_res, as = "parsed", simplifyVector = TRUE)

  #- Download config.json

  cfg_url <- paste0("https://huggingface.co/", model_id,
                    "/raw/main/config.json")

  cfg_res <- httr::GET(

    cfg_url,
    httr::add_headers(Authorization = paste("Bearer", api_key))
  )

  if (httr::status_code(cfg_res) != 200) {

    warning(
      "Failed to retrieve config.json [",
      httr::status_code(cfg_res), "]; skipping save.",
      call. = FALSE
    )

    config <- NULL

  } else {

    config <- jsonlite::fromJSON(rawToChar(cfg_res$content))

    jsonlite::write_json(config, config_path, pretty = TRUE, auto_unbox = TRUE)
  }

  list(

    inference   = inference,
    config      = config,
    config_path = config_path
  )
}

#' Create an nn_module Constructor for HF Feature-Extraction
#'
#' @param model_id HF repo ID to load feature-extraction from.
#'
#' @param api_key API token for HF (default from Sys.getenv("HF_API_KEY")).
#'
#' @return A constructor function for an nn_module named "hf_base".
#'
#' @examples
#'
#' \dontrun{
#' # Retrieve constructor (no inference yet)
#'
#' hf_ctor <- scorch_hf_base_module("distilbert-base-uncased")
#'
#' # Instantiate and get dummy output
#'
#' mod <- hf_ctor()
#'
#' out <- mod()  # dummy tensor
#' }
#'
#' @export

scorch_hf_base_module <- function(

    model_id,
    api_key = Sys.getenv("HF_API_KEY")) {

    res <- scorch_hf_call(

        model_id,
        input = list(inputs = ""),
        api_key = api_key,
        config_path = tempfile("hfconfig", fileext = ".json")
    )

    hidden_size <- res$config$hidden_size

    torch::nn_module(

        "hf_base",

        initialize = function() {
            self$config      <- res$config
            self$model_id    <- model_id
            self$api_key     <- api_key
            self$inference_fn <- res$inference
        },

        forward = function(input_ids) {

            # call the HF inference endpoint, returns logits or embeddings
            out <- self$inference_fn(inputs = input_ids)
            # suppose out is a list with 'last_hidden_state' as a matrix [batch, seq, hidden]
            # here we take the CLS token embedding: out$last_hidden_state[,1,]
            torch_tensor(do.call(rbind, out$last_hidden_state))$slice(2, 1, 1)  # (batch,hidden_size)
        }
    )
}
