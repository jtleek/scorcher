dlv = function(dl,...){
  l = list(dl=dl,
           dl_var=list(...),
           scorch_architecture=list())
  create_scorch_model_class(l)
}


create_scorch_model_class = function(obj) {
  structure(obj, class = "scorch_model")
}

