scorch_function = function(scorch_model,func,...){

  func_call = function(x){
    return(func(x,...))
  }

  scorch_model$scorch_architecture = append(scorch_model$scorch_architecture,
                                            list(func_call,
                                                 type="function"))
  scorch_model

}
