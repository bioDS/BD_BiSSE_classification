# Written using ChatGPT
suppressMessages(library(diversitree))
suppressMessages(library(ape))
get_mle_label <- function(newick_str) {
    phy <- read.tree(text = newick_str)

    lik_biSSE <- make.bisse(phy, states = setNames(rep(NA, length(phy$tip.label)), phy$tip.label), strict = FALSE)
    lik_BD <- constrain(lik_biSSE, lambda1 ~ lambda0, mu1 ~ mu0, q01 ~ 0, q10 ~ 0)
    p_BD_start <- c(lambda1 = 0.5, mu1 = 0.1)
    p_BiSSE_start <- c(lambda1 = 0.5, lambda2 = 0.5, mu1 = 0.1, mu2 = 0.1, q01 = 0.05, q10 = 0.05)
    mle_BD <- find.mle(lik_BD, p_BD_start)
    mle_BiSSE <- find.mle(lik_biSSE, p_BiSSE_start)

    LR <- 2 * (mle_BiSSE$lnLik - mle_BD$lnLik)
    p_val <- 0.5 * pchisq(LR, df = 1, lower.tail = FALSE)
    # old df: length(mle_BiSSE$par) - length(mle_BD$par)
    return(p_val)
    # if (p_val < 0.1) { # 0.05
    #     return(1)
    # }
    # return(0)
}
