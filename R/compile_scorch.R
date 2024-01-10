#' Compile a scorch model
#'
#' @param sm A scorch model archecture
#'
#' @return A nn_module ready for training
#'
#' @export
#'
#' @examples
#'
#' input = mtcars |> as.matrix()
#' output = mtcars |> as.matrix()
#' dl = create_dataloader(input,output,batch_size=2)
#' model = dl |> initiate_scorch() |>
#'   scorch_layer(nn_linear(11,5)) |>
#'   scorch_layer(nn_linear(5,2)) |>
#'   compile_scorch()


compile_scorch = function(sm){

  model = nn_module(
    initialize = function(sm){
      n_layer = length(sm$scorch_architecture)/2
      modules =
        nn_module_list(
          sm$scorch_architecture[seq(1,n_layer*2,by=2)]
        )
      self$modules = modules
    },
    forward = function(input){
      n_layer = length(sm$scorch_architecture)/2
      output = input
      for(i in 1:n_layer){
        output = self$modules[[i]](output)
      }
      return(output)
    }
  )

 sm |> model()

}


create_scorch_nn_module_class = function(model){
    tmp = model
    class(tmp) = c("scorch_nn_module",class(model))
    return(tmp)
}

print.scorch_nn_module = function(model){
  output = capture.output(model$children$modules)
  n_param = stringr::str_extract(output[1],"\\d+")
  n_params_layer = gsub("#","",stringr::str_extract(
    output[grep("#",output)],
    "#\\d+"))
  cat(glue::glue("This scorch_nn_module has {crayon::red(n_param)} parameters \n\n"))
  cat("\n\n")
  for(i in 1:length(n_params_layer)){
    cat(
      glue::glue(" * Layer {i} has {crayon::red(n_params_layer[i])} parameters\n\n"))
  }
}
