## ============================================================
## Comparative Evaluation of Machine Learning Algorithms for
## Predicting Live Body Weight from Morphometric Traits in
## Noiler Chicken (Week 16 data)
##
## Revised pipeline (major revision, Discover Applied Sciences):
##  - identical records grouped: grouped 80/20 split + grouped CV
##  - residual normality, Levene's test, robust ANOVA, Tukey HSD
##  - sensitivity power analysis
## ============================================================

## ------------------------------------------------------------
## 0. SETUP
## ------------------------------------------------------------
# setwd("path/to/your/folder")   # set to the folder holding the CSV

packages <- c("caret", "car", "corrplot", "dplyr", "psych", "nnet", "gbm",
              "randomForest", "kernlab", "pwr")
installed <- packages %in% rownames(installed.packages())
if (any(!installed)) install.packages(packages[!installed])

library(caret); library(car); library(corrplot); library(dplyr)
library(psych); library(nnet); library(gbm); library(pwr)

## ------------------------------------------------------------
## 1. DATA IMPORT AND IDENTICAL-RECORD GROUPING
## ------------------------------------------------------------
df <- read.csv("noiler_week16_data.csv", stringsAsFactors = TRUE)
str(df); sum(is.na(df))   # 0

# Records with identical genotype, sex and all five traits share an ID
key <- do.call(paste, df[, c("genotype", "SEX", "BW16", "BG16", "BL16", "KL16", "SL16")])
df$rec_id <- as.integer(factor(key))
nrow(df)                     # 240
length(unique(df$rec_id))    # 128 distinct records
table(table(df$rec_id))      # 50 x1, 50 x2, 25 x3, 3 x5
table(df$genotype, df$SEX)   # 30 per group

## ------------------------------------------------------------
## 2. DESCRIPTIVE STATISTICS
## ------------------------------------------------------------
traits <- c("BW16", "BG16", "BL16", "KL16", "SL16")
describe(df[, traits])

# Table 2: N, mean, SE, SD, CV (%), min, max
table2 <- data.frame(
  Trait = traits,
  N     = sapply(df[, traits], length),
  Mean  = sapply(df[, traits], mean),
  SE    = sapply(df[, traits], function(x) sd(x) / sqrt(length(x))),
  SD    = sapply(df[, traits], sd),
  CV    = sapply(df[, traits], function(x) sd(x) / mean(x) * 100),
  Min   = sapply(df[, traits], min),
  Max   = sapply(df[, traits], max))
print(table2, digits = 4)
write.csv(table2, "Table2_descriptives.csv", row.names = FALSE)

# Table 3: BW16 mean and SD by genotype x sex, genotype and sex
df %>% group_by(genotype, SEX) %>% summarise(mean = mean(BW16), sd = sd(BW16), .groups = "drop")
df %>% group_by(genotype) %>% summarise(mean = mean(BW16), sd = sd(BW16))
df %>% group_by(SEX) %>% summarise(mean = mean(BW16), sd = sd(BW16))

# Figure 1 and Figures 3a/3b
png("Fig1_histogram_BW16.png", width = 1800, height = 1350, res = 300)
hist(df$BW16, main = "Distribution of Body Weight (Week 16)", xlab = "BW16 (g)", col = "lightblue")
dev.off()
png("Fig3a_boxplot_genotype.png", width = 1800, height = 1350, res = 300)
boxplot(BW16 ~ genotype, data = df, main = "BW16 by Genotype", col = "lightgreen")
dev.off()
png("Fig3b_boxplot_sex.png", width = 1800, height = 1350, res = 300)
boxplot(BW16 ~ SEX, data = df, main = "BW16 by Sex", col = "lightpink")
dev.off()

## ------------------------------------------------------------
## 3. ANOVA AND ASSUMPTION CHECKS
## ------------------------------------------------------------
anova_model <- aov(BW16 ~ genotype * SEX, data = df)
summary(anova_model)

shapiro.test(residuals(anova_model))                               # normality of residuals
png("Fig2_QQplot_residuals.png", width = 1800, height = 1350, res = 300)
qqnorm(residuals(anova_model), main = "Normal Q-Q plot of ANOVA residuals")
qqline(residuals(anova_model), col = "red")
dev.off()

leveneTest(BW16 ~ genotype * SEX, data = df, center = median)     # homogeneity of variance
Anova(lm(BW16 ~ genotype * SEX, data = df), type = 3, white.adjust = TRUE)  # robust ANOVA

TukeyHSD(anova_model, "genotype")
TukeyHSD(anova_model, "genotype:SEX")

# Check on distinct records only
df_unique <- df[!duplicated(df$rec_id), ]
summary(aov(BW16 ~ genotype * SEX, data = df_unique))

## ------------------------------------------------------------
## 4. CORRELATION AND MULTICOLLINEARITY
## ------------------------------------------------------------
corr_matrix <- cor(df[, traits]); print(corr_matrix)
png("Fig4_correlation_plot.png", width = 1800, height = 1800, res = 300)
corrplot(corr_matrix, method = "number", type = "upper")
dev.off()
vif(lm(BW16 ~ BG16 + BL16 + KL16 + SL16, data = df))

## ------------------------------------------------------------
## 5. SENSITIVITY POWER ANALYSIS
## ------------------------------------------------------------
pwr.f2.test(u = 3, v = 232, sig.level = 0.05, power = 0.80)   # genotype, genotype x sex
pwr.f2.test(u = 1, v = 232, sig.level = 0.05, power = 0.80)   # sex
pwr.f2.test(u = 4, v = 189, sig.level = 0.05, power = 0.80)   # 4-predictor model, training set
ss  <- summary(anova_model)[[1]][["Sum Sq"]]
eta <- ss[1:3] / (ss[1:3] + ss[4]); names(eta) <- c("genotype", "SEX", "GxS")
round(eta, 3)
pwr.f2.test(u = 3, v = 232, f2 = eta["GxS"] / (1 - eta["GxS"]), sig.level = 0.05)$power

## ------------------------------------------------------------
## 6. GROUPED 80/20 SPLIT (identical records kept together)
## ------------------------------------------------------------
set.seed(123)
grp    <- df[!duplicated(df$rec_id), c("rec_id", "BW16")]
tr_ids <- grp$rec_id[createDataPartition(grp$BW16, p = 0.8, list = FALSE)]
train_data <- df[df$rec_id %in% tr_ids, ]
test_data  <- df[!(df$rec_id %in% tr_ids), ]
nrow(train_data); nrow(test_data)                          # 194 / 46
length(intersect(train_data$rec_id, test_data$rec_id))     # 0
table(test_data$genotype, test_data$SEX)

## ------------------------------------------------------------
## 7. GROUPED 10-FOLD CV, REPEATED 3 TIMES (shared by all models)
## ------------------------------------------------------------
set.seed(123)
folds <- c(groupKFold(train_data$rec_id, k = 10),
           groupKFold(train_data$rec_id, k = 10),
           groupKFold(train_data$rec_id, k = 10))
names(folds) <- paste0("Fold", sprintf("%02d", 1:10), ".Rep", rep(1:3, each = 10))
train_control <- trainControl(method = "cv", index = folds, savePredictions = "final")

f <- BW16 ~ BG16 + BL16 + KL16 + SL16

## ------------------------------------------------------------
## 8. MODELS
## ------------------------------------------------------------
set.seed(123); model_mlr <- train(f, data = train_data, method = "lm", trControl = train_control)
set.seed(123); model_rf  <- train(f, data = train_data, method = "rf", trControl = train_control, importance = TRUE)
set.seed(123); model_svm <- train(f, data = train_data, method = "svmRadial", trControl = train_control,
                                  preProcess = c("center", "scale"))
set.seed(123); model_ann <- train(f, data = train_data, method = "nnet", trControl = train_control,
                                  preProcess = c("center", "scale"), linout = TRUE, trace = FALSE)
set.seed(123); model_gbm <- train(f, data = train_data, method = "gbm", trControl = train_control, verbose = FALSE)

models <- list(MLR = model_mlr, RF = model_rf, SVM = model_svm, ANN = model_ann, GBM = model_gbm)
lapply(models, function(m) m$bestTune)

## ------------------------------------------------------------
## 9. MODEL COMPARISON (cross-validated resamples)
## ------------------------------------------------------------
results <- resamples(models)
summary(results)
summary(diff(results))        # paired t-tests, Bonferroni-adjusted

png("Fig5a_RMSE.png", width = 1800, height = 1200, res = 300)
print(dotplot(results, metric = "RMSE")); dev.off()
png("Fig5b_R2.png", width = 1800, height = 1200, res = 300)
print(bwplot(results, metric = "Rsquared")); dev.off()

## ------------------------------------------------------------
## 10. TEST-SET VALIDATION AND VARIABLE IMPORTANCE
## ------------------------------------------------------------
test_tab <- t(sapply(models, function(m) postResample(predict(m, test_data), test_data$BW16)))
round(test_tab, 3)
lapply(models, varImp)

png("Fig6_diagnostics.png", width = 4500, height = 1900, res = 300)
par(mfrow = c(2, 5), mar = c(4.2, 4.2, 2.5, 1))
for (nm in names(models)) {
  p <- predict(models[[nm]], test_data)
  plot(test_data$BW16, p, xlab = "Observed BW (g)", ylab = "Predicted BW (g)",
       main = nm, pch = 19, xlim = c(1300, 3000), ylim = c(1300, 3000))
  abline(0, 1, col = "red", lwd = 2)
}
for (nm in names(models)) {
  p <- predict(models[[nm]], test_data)
  plot(p, test_data$BW16 - p, xlab = "Predicted BW (g)", ylab = "Residual (g)", main = nm, pch = 19)
  abline(h = 0, col = "red", lwd = 2)
}
dev.off()

## ------------------------------------------------------------
## 11. SAVE MODELS + SESSION INFO
## ------------------------------------------------------------
for (nm in names(models)) saveRDS(models[[nm]], paste0("model_", tolower(nm), ".rds"))
sessionInfo()
