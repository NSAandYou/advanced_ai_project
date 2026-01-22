model1 <- c(0.81, 0.83, 0.79, 0.84, 0.80)
model2 <- c(0.76, 0.78, 0.77, 0.75, 0.79)
model3 <- c(0.89, 0.88, 0.87, 0.90, 0.91)

data <- data.frame(model1, model2, model3)

# Friedman test
print(friedman.test(as.matrix(data)))

# Pairwise Wilcoxon
wilcox_res <- pairwise.wilcox.test(
  x = unlist(data),
  g = rep(colnames(data), each = nrow(data)),
  paired = TRUE,
  p.adjust.method = "holm"
)

print(wilcox_res)
