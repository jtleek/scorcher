#' Compile a scorch model
#'
#' @param sm A scorch model architecture
#'
#' @return A scorch_model object with components (1) nn_module compiled and ready to train and (2) the scorch_dataloader object
#'
#' @import torch
#'
#' @examples
#'
#' input = mtcars |> as.matrix() |> torch::torch_tensor()
#' output = mtcars |> as.matrix() |> torch::torch_tensor()
#' dl = create_dataloader(input,output,batch_size=2)
#' scorch_model = dl |> initiate_scorch() |>
#'   scorch_layer(torch::nn_linear(11,5)) |>
#'   scorch_layer(torch::nn_linear(5,2)) |>
#'   compile_scorch()
#'
#' @export

compile_scorch = function(sm){

  model = nn_module(
    initialize = function(sm){
      n_layer = length(sm$scorch_architecture)/2
      layer_types = sm$scorch_architecture[2*(1:n_layer)] |> unlist()
      layer_index = which(layer_types == "layer")*2-1
      func_index = which(layer_types == "function")*2-1
      modules =
        nn_module_list(
          sm$scorch_architecture[layer_index]
        )
      self$modules = modules
      self$functions = sm$scorch_architecture[func_index]
    },
    forward = function(input){
      n_layer = length(sm$scorch_architecture)/2
      layer_types = sm$scorch_architecture[2*(1:n_layer)] |> unlist()
      layer_index = which(layer_types == "layer")
      output = input
      i_layer = i_function = 1
      for(i in 1:n_layer){
        if(i %in% layer_index){
          output = self$modules[[i_layer]](output)
          i_layer = i_layer + 1
        }else{
          output = self$functions[[i_function]](output)
          i_function. = i_function+1
        }
      }
      return(output)
    }
  )


 return(list(nn_model = sm |> model(), dl = sm$dl))

}

#' Create a Compiled Scorch Model Class
#'
#' This function creates an object of class 'compiled_scorch_model' by appending it to the existing classes of the input object.
#'
#' @param obj An object to be converted to a compiled scorch model.
#'
#' @return The input object with the class attribute set to include 'compiled_scorch_model'.
#' @export
#'
#' @examples
#' model <- list(children = list(modules = "module details"))
#' compiled_model <- create_scorch_nn_module_class(model)
#' class(compiled_model)

create_scorch_nn_module_class = function(obj){
    class(model) = c("compiled_scorch_model",class(model))
    return(model)
}

#' Print Method for Scorch Neural Network Module
#'
#' This function defines the print method for objects of class 'scorch_nn_module'.
#'
#' @param x An object of class 'scorch_nn_module'.
#'
#' @param ... Additional arguments to be passed to the print function.
#'
#' @export
#'
#' @examples
#' model <- list(children = list(modules = "module details"))
#' model$children$modules <- "modules: 123\nlayer1 #64\nlayer2 #32"
#' class(model) <- c("scorch_nn_module", class(model))
#' print(model)

print.scorch_nn_module = function(x, ...){
  output = capture.output(x$children$modules)
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


apply_layer = function(modules,functions,layer_index,n_layer){

}
