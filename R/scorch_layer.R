

scorch_layer = function(scorch_model,nn_obj,...){

  l = list(...)
  scorch_model$scorch_architecture = append(scorch_model$scorch_architecture,
                                            list(nn_obj,l))
  scorch_model

}


