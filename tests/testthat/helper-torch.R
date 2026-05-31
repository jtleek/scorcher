skip_if_no_torch_backend <- function() {
  testthat::skip_if_not_installed("torch")

  ok <- tryCatch({
    torch::torch_empty(1)
    TRUE
  }, error = function(e) FALSE)

  testthat::skip_if_not(ok, "torch backend is not installed")
}

