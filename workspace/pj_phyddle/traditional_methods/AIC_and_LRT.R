# Use AIC and LRT to perform birth-death model classification.
# Author: Kate Truman, with the use of AI tools.

library("diversitree")
library("ape")
library(foreach)
library(doParallel)
library("rlist")
library("ggplot2")
library("dplyr")


setwd("/home/ket581/AIphylo/phyddle/workspace/pj_phyddle/traditional_model_selection_thesis")
file.remove("parallel_log_new_indices_changed_start.txt")


true_test_labels <- read.csv("test_ids.csv", header=T)
colnames(true_test_labels) <- c("ID", "tree_path","true_label")
true_test_labels$true_label <- as.factor(true_test_labels$true_label)
bisse_test_indices = true_test_labels$ID[true_test_labels$ID >= 1000000]
bd_test_indices = true_test_labels$ID[true_test_labels$ID < 1000000]

cluster <- makeCluster(80)
registerDoParallel(cluster)
clusterEvalQ(cluster, library(diversitree))
indices = true_test_labels$ID
n = length(indices)
clusterExport(cluster, varlist = c("indices", "n"))

results <- foreach(ind = 1:(length(indices)), .packages = "diversitree") %dopar% { # 
tryCatch({
  adj_ind = ind
  print(indices[adj_ind])
  tree= read.tree(file=paste0("/mnt/DiversificationGraphInference/thirty_thousand_trees/simulate/out.", indices[adj_ind], ".downsampled.tre")) 
  params = list()
  vals = starting.point.bisse(tree)
  cat("task ", ind, "\n", file="parallel_log.txt", append=TRUE)
  states = setNames(rep(NA, length(tree$tip.label)),tree$tip.label)
  bisse_model = make.bisse(tree, states, strict=FALSE)
  constrained_bisse = constrain(bisse_model, q01 ~ q10) #, q10 ~ (lambda0 + lambda1) / 20)
  constrain_bd_from_bisse = constrain(bisse_model, q01 ~ 0, q10 ~ 0, lambda0 ~ lambda1, mu0 ~ mu1)
  vals_short = setNames(starting.point.bd(tree), c("lambda0", "mu0"))
  lower_vec = c(
        lambda0 = 0.1,
        lambda1 = 0.1,
        mu0 = 0,
        mu1 = 0,
        q01 = 0.01
    )
upper_vec = c(
        lambda0 = 1,
        lambda1 = 1,
        mu0 = 0.9,
        mu1 = 0.9,
        q01 = 0.1
    )
    manual_vals = vals[1:5]  
    manual_vals[1] = max(manual_vals[1] * 0.75, 0.1)
    manual_vals[2] = min(manual_vals[2] / 0.75, 1)
    manual_vals[3] = max(manual_vals[3] * 0.75, 0.1)
    manual_vals[4] = min(manual_vals[4] / 0.75, 0.9)
    minimum_vals = vals[1:5]

    minimum_vals[1] = 0.1
    minimum_vals[2] = 0.1
    minimum_vals[3] = 0
    minimum_vals[4] = 0
    minimum_vals[5] = 0.05

    maximum_vals = vals[1:5]
    maximum_vals[1] = 1
    maximum_vals[2] = 1
    maximum_vals[3] = 0.9
    maximum_vals[4] = 0.9
    maximum_vals[5] = 0.05

    manual_vals_short = vals_short
    manual_vals_short[1] = max(vals_short[1] * 0.75, 0.1)
    manual_vals_short[2]  = max(vals_short[2] * 0.75, 0.1)

    minimum_vals_short = c(minimum_vals[1], minimum_vals[3])
    maximum_vals_short = c(maximum_vals[1], maximum_vals[3])

    bd_mle_1 = find.mle(constrain_bd_from_bisse, vals_short, 
                lower = c(lambda=0.1, mu=0),
                upper=c(lambda=1,mu=0.9))
    bd_mle_2 = find.mle(constrain_bd_from_bisse, manual_vals_short, 
                lower = c(lambda=0.1, mu=0),
                upper=c(lambda=1,mu=0.9))
    bd_mle_3 = find.mle(constrain_bd_from_bisse, minimum_vals_short, 
                lower = c(lambda=0.1, mu=0),
                upper=c(lambda=1,mu=0.9))
    bd_mle_4 = find.mle(constrain_bd_from_bisse, maximum_vals_short, 
                lower = c(lambda=0.1, mu=0),
                upper=c(lambda=1,mu=0.9))
    
  bisse_mle_1 = find.mle(constrained_bisse, vals[1:5], lower = lower_vec, upper = upper_vec)
  bisse_mle_2 = find.mle(constrained_bisse, manual_vals, lower = lower_vec, upper = upper_vec)
bisse_mle_3 = find.mle(constrained_bisse, minimum_vals, lower = lower_vec, upper = upper_vec)
  bisse_mle_4 = find.mle(constrained_bisse, maximum_vals, lower = lower_vec, upper = upper_vec)

    bd_mles <- list(
        bd_mle_1,
        bd_mle_2,
        bd_mle_3,
        bd_mle_4
    )

    bisse_mles <- list(
        bisse_mle_1,
        bisse_mle_2,
        bisse_mle_3,
        bisse_mle_4
    )
    # Remove parameter sets found under BiSSE which are too similar to BD
    for (m in seq_along(bisse_mles)){
        print(m)
        pars = bisse_mles[[m]]$par
        ratio_1 = pars[[1]] / pars[[2]]
        if (ratio_1 < 1){
            ratio_1 = 1 / ratio_1
        }
        ratio_2 = pars[[3]] / pars[[4]]
        if (ratio_2 < 1){
            ratio_2 = 1 / ratio_2
        }
        if (ratio_1 < 1.1 && ratio_2 < 1.1){
            bisse_mles[[m]]$lnLik = NA
        }
    }

    bd_log_liks <- sapply(bd_mles, function(x) x$lnLik)
    best_bd_mle <- bd_mles[[which.max(bd_log_liks)]]
    bisse_log_liks <- sapply(bisse_mles, function(x) x$lnLik)
    best_bisse_mle <- bisse_mles[[which.max(bisse_log_liks)]]
  return(list(indices[adj_ind], anova(constrained=best_bd_mle, best_bisse_mle), best_bd_mle$par, best_bisse_mle$par))
}, error = function(e) {
    message("An error occurred: ", e$message)
    return(NA)
})
}
stopCluster(cl = cluster)
#saveRDS(results, "results.rds")
#write.csv(results,"results.csv")

formatted_results <- lapply(results[!vapply(results, function(x) length(x) == 1 && is.na(x), logical(1))], function(x) {

    idx <- x[[1]]
    anova_result <- x[[2]]
    bd_par <- x[[3]]
    bisse_par <- x[[4]]

    data.frame(
        idx = idx,

        bisse_lnLik = anova_result["full", "lnLik"],
        bisse_AIC = anova_result["full", "AIC"],

        bd_lnLik = anova_result["constrained", "lnLik"],
        bd_AIC = anova_result["constrained", "AIC"],
        chisq = anova_result["constrained", "ChiSq"],
        p_val = anova_result["constrained", "Pr(>|Chi|)"],
        fit_bd_lambda = bd_par[[1]],
        fit_bd_mu = bd_par[[2]],
        fit_bisse_lambda0 = bisse_par[[1]],
        fit_bisse_lambda1 = bisse_par[[2]],
        fit_bisse_mu0 = bisse_par[[3]],
        fit_bisse_mu1 = bisse_par[[4]],
        fit_bisse_q = (bisse_par[[1]] + bisse_par[[2]])/20
    )
})

formatted_results <- do.call(rbind, formatted_results)
saveRDS(formatted_results, "results_new_indices_changed_start.rds")
write.csv(formatted_results,"results_new_indices_changed_start.csv", row.names=FALSE, quote = FALSE)

