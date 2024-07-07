

scorch_layer = function(scorch_model,nn_obj){

  scorch_model$scorch_architecture = append(scorch_model$scorch_architecture,
                                            list(nn_obj,
                                                 type="layer"))
  scorch_model

}


