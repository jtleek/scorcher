

scorch_layer = function(scorch_model,nn_obj,...){

#  stopifnot("`nn_layer` must be a nn_module_generator (a nn layer type from torch)" = "nn_module_generator" %in% class(nn_layer))

  l = list(...)
  scorch_model$scorch_architecture = append(scorch_model$scorch_architecture,list(nn_obj,l))
  scorch_model

}

