args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 9) {
  stop("Expected 9 arguments: input_dir output_dir survival_upper d_num_pts warmup iterations chains parallel seed")
}

input_dir <- args[[1]]
output_dir <- args[[2]]
ls_est <- as.numeric(args[[3]])
d_num_pts <- as.integer(args[[4]])
warmup <- as.integer(args[[5]])
iterations <- as.integer(args[[6]])
chains <- as.integer(args[[7]])
parallel <- as.integer(args[[8]])
seed <- as.integer(args[[9]])

if (!requireNamespace("EpiPvr", quietly = TRUE)) {
  stop("R package 'EpiPvr' is not installed.")
}
if (!requireNamespace("posterior", quietly = TRUE)) {
  stop("R package 'posterior' is not installed; it should be installed with EpiPvr.")
}

set.seed(seed)

read_assay <- function(path) {
  frame <- read.csv(path, check.names = FALSE)
  required <- c("T_vec", "R_vec", "I_vec")
  if (!all(required %in% names(frame))) {
    stop(paste("Assay file missing columns:", path))
  }
  result <- rbind(
    T_vec = as.numeric(frame$T_vec),
    R_vec = as.numeric(frame$R_vec),
    I_vec = as.numeric(frame$I_vec)
  )
  return(result)
}

metadata <- read.csv(file.path(input_dir, "metadata.csv"), stringsAsFactors = FALSE)
mode <- toupper(as.character(metadata$transmission_type[[1]]))
vectors_per_plant <- as.integer(metadata$vectors_per_plant[[1]])
durations <- as.matrix(read.csv(file.path(input_dir, "durations.csv"), header = FALSE))
storage.mode(durations) <- "double"

if (mode == "SPT") {
  rownames(durations) <- c("AAPfixedComponent", "IAPfixedComponent")
  data_in <- list(
    d_AAP = read_assay(file.path(input_dir, "AAP.csv")),
    d_IAP = read_assay(file.path(input_dir, "IAP.csv")),
    d_durations = durations,
    d_vectorspp = vectors_per_plant,
    d_virusType = "SPT"
  )
  fit <- EpiPvr::estimate_virus_parameters_SPT(
    data_in,
    ls_est,
    D_numPtsPdin = d_num_pts,
    mcmcOptions = c(warmup, iterations),
    numChainsIn = chains,
    mc.parallel = parallel
  )
} else if (mode == "PT") {
  rownames(durations) <- c("AAPfixedComponent", "LAPfixedComponent", "IAPfixedComponent")
  data_in <- list(
    d_AAP = read_assay(file.path(input_dir, "AAP.csv")),
    d_LAP = read_assay(file.path(input_dir, "LAP.csv")),
    d_IAP = read_assay(file.path(input_dir, "IAP.csv")),
    d_durations = durations,
    d_vectorspp = vectors_per_plant,
    d_virusType = "PT"
  )
  fit <- EpiPvr::estimate_virus_parameters_PT(
    data_in,
    ls_est,
    D_numPtsPdin = d_num_pts,
    mcmcOptions = c(warmup, iterations),
    numChainsIn = chains,
    mc.parallel = parallel
  )
} else {
  stop(paste("Unsupported transmission type:", mode))
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

draws <- posterior::as_draws_df(fit$array1)
find_column <- function(candidates) {
  for (candidate in candidates) {
    if (candidate %in% names(draws)) return(candidate)
  }
  stop(paste("Could not find posterior variable among:", paste(candidates, collapse = ", ")))
}

al_col <- find_column(c("al[1]", "al"))
be_col <- find_column(c("be[1]", "be"))
mu_col <- find_column(c("mu[1]", "mu"))

posterior_out <- data.frame(
  acquisition_rate = as.numeric(draws[[al_col]]),
  inoculation_rate = as.numeric(draws[[be_col]]),
  vector_clearance_rate = as.numeric(draws[[mu_col]])
)

if (mode == "PT") {
  lat_col <- find_column(c("lat[1]", "lat"))
  posterior_out$vector_latent_progression_rate <- as.numeric(draws[[lat_col]])
}

if (".chain" %in% names(draws)) posterior_out$chain <- as.integer(draws$.chain)
if (".iteration" %in% names(draws)) posterior_out$iteration <- as.integer(draws$.iteration)
if (".draw" %in% names(draws)) posterior_out$draw <- as.integer(draws$.draw)

write.csv(posterior_out, file.path(output_dir, "posterior.csv"), row.names = FALSE)
write.csv(as.data.frame(fit$array2), file.path(output_dir, "summary.csv"), row.names = FALSE)

diag <- fit$converge_results
diagnostics <- data.frame(
  key = c("divergent_transitions", "max_treedepth_exceeded"),
  value = c(
    paste(diag$divergent_transitions, collapse = ";"),
    paste(diag$max_treedepth_exceeded, collapse = ";")
  )
)
write.csv(diagnostics, file.path(output_dir, "diagnostics.csv"), row.names = FALSE)

# In EpiPvr 0.0.1 the Bayesian-R2 object is array5 for SPT and array6 for PT.
bayes_object <- if (mode == "SPT") fit$array5 else fit$array6
if (!is.null(bayes_object) && !is.null(bayes_object$bayesR2_mn)) {
  assay_names <- if (mode == "SPT") c("AAP", "IAP") else c("AAP", "LAP", "IAP")
  n_values <- length(bayes_object$bayesR2_mn)
  use_n <- min(length(assay_names), n_values)
  bayes <- data.frame(
    assay = assay_names[seq_len(use_n)],
    mean = as.numeric(bayes_object$bayesR2_mn[seq_len(use_n)]),
    sd = as.numeric(bayes_object$bayesR2_sd[seq_len(use_n)])
  )
  write.csv(bayes, file.path(output_dir, "bayes_r2.csv"), row.names = FALSE)
}

capture.output(sessionInfo(), file = file.path(output_dir, "R_session_info.txt"))
