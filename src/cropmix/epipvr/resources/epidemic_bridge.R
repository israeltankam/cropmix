args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 11) {
  stop("Expected 11 arguments: output num_insects interval alpha beta mu dispersal roguing harvest mortality plant_latent")
}
if (!requireNamespace("EpiPvr", quietly = TRUE)) {
  stop("R package 'EpiPvr' is not installed.")
}

output <- args[[1]]
num_insects <- as.integer(args[[2]])
interval <- as.numeric(args[[3]])
alpha <- as.numeric(args[[4]])
beta <- as.numeric(args[[5]])
mu <- as.numeric(args[[6]])
dispersal <- as.numeric(args[[7]])
roguing <- as.numeric(args[[8]])
harvest <- as.numeric(args[[9]])
vector_mortality <- as.numeric(args[[10]])
plant_latent <- as.numeric(args[[11]])

# Semantic ordering follows the EpiPvr vignette:
# dispersal, roguing, harvesting, vector mortality, plant latent progression.
local_params <- c(dispersal, roguing, harvest, vector_mortality, plant_latent)
virus_params <- c(alpha, beta, mu)
probabilities <- EpiPvr::calculate_epidemic_probability(
  num_insects,
  interval,
  local_params,
  virus_params
)
write.csv(data.frame(probability = as.numeric(probabilities)), output, row.names = FALSE)
