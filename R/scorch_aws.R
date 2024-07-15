#' Upload a Scorch Model to AWS, Train on EC2, and Retrieve Results
#'
#' @description
#' Automates the process of uploading a scorch model to AWS S3, launching an
#' EC2 instance to train, and retrieving the results. Includes safeguards
#' to prevent unexpected costs by requiring double confirmation, limiting file
#' size, and setting an automatic termination time for the EC2 instance.
#'
#' @param scorch_model A scorch model object to be uploaded and trained.
#'
#' @param testing_data A torch tensor containing the data for predictions.
#'
#' @param loss A loss function from the torch package
#'
#' @param loss_params Parameters to pass to the loss function
#'
#' @param optim An optimizer from the torch package
#'
#' @param optim_params Parameters to pass to the optimizer
#'
#' @param num_epochs The number of epochs to train for
#'
#' @param s3_bucket The name of the S3 bucket where the model and results will
#' be stored.
#'
#' @param s3_model_key The key (filename) for the model in S3.
#' Default is "model.pth".
#'
#' @param s3_result_key The key (filename) for the results in S3.
#' Default is "results.pth".
#'
#' @param ec2_image_id The AMI ID for the EC2 instance.
#'
#' @param ec2_instance_type The type of EC2 instance to launch.
#' Default is "t2.micro".
#'
#' @param ec2_key_name The name of the key pair for SSH access to the
#' EC2 instance.
#'
#' @param ec2_security_group The name of the security group for the
#' EC2 instance.
#'
#' @param ec2_region The AWS region where the EC2 instance will be launched.
#' Default is "us-east-1".
#'
#' @param ec2_script_path The local path to save the EC2 processing script.
#' Default is "train_scorch_model.R".
#'
#' @param max_file_size_mb The maximum allowed file size for upload in MB.
#' Default is 100.
#'
#' @param max_run_time_minutes The maximum allowed runtime for the EC2 instance
#' in minutes. Default is 60.
#'
#' @param ... Additional arguments passed to `compile_scorch` and `fit_scorch`.
#'
#' @return The results of the model fitting as a torch tensor.
#'
#' @export
#'
#' @examples
#'
#' \dontrun{
#' input  <- mtcars |> as.matrix() |> torch::torch_tensor()
#'
#' output <- mtcars |> as.matrix() |> torch::torch_tensor()
#'
#' dl <- scorch_create_dataloader(input, output, batch_size = 2)
#'
#' scorch_model <- dl |> initiate_scorch() |>
#'
#'   scorch_layer(torch::nn_linear(11,5)) |>
#'
#'   scorch_layer(torch::nn_linear(5,2)) |>
#'
#'   scorch_layer(torch::nn_linear(2,5)) |>
#'
#'   scorch_layer(torch::nn_linear(5,11))
#'
#' results <- scorch_to_aws(
#'
#'   scorch_model = scorch_model,
#'
#'   testing_data = output,
#'
#'   s3_bucket = "your-bucket-name",
#'
#'   ec2_image_id = "ami-xxxxxxxx",
#'
#'   ec2_key_name = "your-key-pair",
#'
#'   ec2_security_group = "your-security-group"
#' )
#'
#' print(results)
#' }

scorch_to_aws <- function(scorch_model, testing_data, loss = nn_mse_loss,

  loss_params = list(reduction = "mean"), optim = optim_adam,

  optim_params = list(lr = 0.001), num_epochs = 10,

  s3_bucket, s3_model_key = "model.pth",

  s3_result_key = "results.pth", ec2_image_id, ec2_instance_type = "t2.micro",

  ec2_key_name, ec2_security_group, ec2_region = "us-east-1",

  ec2_script_path = "train_scorch_model.R",

  max_file_size_mb = 100, max_run_time_minutes = 60, ...) {

  ## Double Confirmation before Execution

  cat("WARNING: This operation will incur AWS charges.\n\n",
      "Please confirm to proceed (Y/N): ")

  response1 <- readline()

  if (tolower(response1) != "y") stop("Operation aborted by the user.")

  cat("Are you sure you want to proceed? This will incur AWS charges (Y/N): ")

  response2 <- readline()

  if (tolower(response2) != "y") stop("Operation aborted by the user.")

  ## Save Model Locally

  torch::torch_save(scorch_model, 'model.pth')

  ## Check File Size

  model_size_mb <- file.info('model.pth')$size / (1024^2)

  if (model_size_mb > max_file_size_mb) {

    stop(paste("File size exceeds the maximum allowed size of",
               max_file_size_mb, "MB."))
  }

  ## Display AWS Billing Dashboard Link

  cat("Check your AWS billing dashboard for real-time monitoring:\n\n",
      "https://console.aws.amazon.com/billing/home\n")

  ## Set AWS Credentials

  Sys.setenv("AWS_DEFAULT_REGION" = ec2_region)

  ## Upload Model to S3

  aws.s3::put_object(file = "model.pth",

    object = s3_model_key, bucket = s3_bucket)

  ## Launch EC2 Instance

  instance_info <- aws.ec2::run_instances(image_id = ec2_image_id,

    instance_type = ec2_instance_type, key_name = ec2_key_name,

    security_groups = ec2_security_group, region = ec2_region)

  instance_id <- instance_info$instancesSet[[1]]$instanceId

  ## Set a Timer to Automatically Stop EC2 Instance

  cat("EC2 instance will automatically terminate after",
      max_run_time_minutes, "minutes.\n")

  terminate_ec2 <- function() {

    aws.ec2::terminate_instances(instance_ids = instance_id,

      region = ec2_region)

    cat("EC2 instance terminated.\n")
  }

  on.exit(terminate_ec2(), add = TRUE)

  ## Wait for Instance to Run

  aws.ec2::ec2_wait(instance_id, status = "running", region = ec2_region)

  ## Get Public DNS of Instance

  instance_desc <- aws.ec2::ec2_describe_instances(

    instance_ids = instance_id, region = ec2_region)

  public_dns <- instance_desc$reservations[[1]]$instances[[1]]$publicDnsName

  ## Connect to Instance via SSH and Run Script

  ssh_session <- ssh::ssh_connect(paste0("ec2-user@", public_dns),

    keyfile = paste0("~/.ssh/", ec2_key_name, ".pem"))

  ## Schedule EC2 Termination

  shutdown_command <- paste0("sudo shutdown -h +", max_run_time_minutes)

  ssh::ssh_exec_wait(ssh_session, shutdown_command)

  cat("EC2 instance scheduled to terminate after",
      max_run_time_minutes, "minutes.\n")

  ## Create Training Script on Instance

  script_content <- paste0("

  library(scorcher)
  library(torch)
  library(aws.s3)

  ## Set AWS Credentials

  Sys.setenv('AWS_DEFAULT_REGION' = '", ec2_region, "')

  ## Download Model from S3

  aws.s3::save_object(object = '", s3_model_key,

    "', bucket = '", s3_bucket, "', file = 'model.pth')

  ## Load Model and Train Model

  scorch_model <- torch::torch_load('model.pth')

  compiled_scorch_model <- scorch_model |>

    compile_scorch('", ..., "')

  fitted_scorch_model <- compiled_scorch_model |>

    fit_scorch(loss ='", loss,"', loss_params ='", loss_params,"',

      num_epochs ='", num_epochs, "', verbose = F,'", ..., "')

  fitted_scorch_model$eval()

  output <- fitted_scorch_model(testing_data)

  ## Save Results

  torch::torch_save(output, 'results.pth')

  ## Upload Results to S3

  aws.s3::put_object(file = 'results.pth',

    object = '", s3_result_key, "', bucket = '", s3_bucket, "')
  ")

  writeLines(script_content, ec2_script_path)

  ssh::scp_upload(ssh_session, ec2_script_path, "train_model.R")

  ssh::ssh_exec_wait(ssh_session, "Rscript train_model.R")

  ## Download Results from S3

  aws.s3::save_object(object = s3_result_key, bucket = s3_bucket,

    file = "results.pth")

  ssh::ssh_disconnect(ssh_session)

  ## Load Results

  results <- torch::torch_load("results.pth")

  return(results)
}

#== END ========================================================================
