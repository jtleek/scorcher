#' Compile a scorch model
#'
#' @param sm A scorch model architecture
#'
#' @return A scorch_model object with components (1) nn_module compiled and ready to train and (2) the scorch_dataloader object
#'
#' @export
#'
#' @examples
#'
#' input = mtcars |> as.matrix()
#' output = mtcars |> as.matrix()
#' dl = create_dataloader(input,output,batch_size=2)
#' scorch_model = dl |> initiate_scorch() |>
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


 return(list(nn_model = sm |> model(), dl = dl))

}


create_scorch_nn_module_class = function(obj){
    class(model) = c("compiled_scorch_model",class(model))
    return(model)
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
