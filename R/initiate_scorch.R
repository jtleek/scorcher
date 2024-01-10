#' Initiate a scorch model
#'
#' @param dl An input data loader, created with create_dataloader
#'
#' @return A scorch model object
#'
#' @export
#'
#' @examples
#'
#' input = mtcars |> as.matrix()
#' output = mtcars |> as.matrix()
#' dl = create_dataloader(input,output,batch_size=2)
#' dl |> initiate_scorch()

initiate_scorch = function(dl){
  l = list(dl=dl,
           scorch_architecture=list())
  create_scorch_model_class(l)
}


create_scorch_model_class = function(obj) {
  structure(obj, class = "scorch_model")
}


print.scorch_model = function(sm){
  cat("This scorch model has a dataloader object with features: \n\n")
  print(dl)
  cat("\n and model architecture:\n\n")

  n_layer = length(sm$scorch_architecture)/2

  for(i in 1:n_layer){
    cat(
      glue::glue(" * Layer {i} is a {crayon::red(class(sm$scorch_architecture[(i*2-1)][[1]])[1])} layer\n\n"))
  }

}

