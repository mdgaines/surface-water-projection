#lmm.r

library(lme4)
library(MuMIn)
library(data.table)
library(sjmisc)
library(lmerTest)
library(stringr)
library(Matrix)

##### IMPORT DATA #####

## Surface Water Data
dswe <- read.csv("../data/all_data_0118_p2.csv", stringsAsFactors = F)

# log transform dswe water
dswe$LOG_PR_WATER <- log(dswe$PR_WATER + 10e-6)

# Center and standardize independent variables
dswe$MAX_TMP <- (dswe$MAX_TMP - mean(dswe$MAX_TMP))/sd(dswe$MAX_TMP)
dswe$PRECIP <- (dswe$PRECIP - mean(dswe$PRECIP))/sd(dswe$PRECIP)

dswe$PR_AG <- (dswe$PR_AG - mean(dswe$PR_AG))/sd(dswe$PR_AG)
dswe$PR_NAT <- (dswe$PR_NAT - mean(dswe$PR_NAT))/sd(dswe$PR_NAT)
dswe$PR_INT <- (dswe$PR_INT - mean(dswe$PR_INT))/sd(dswe$PR_INT)

# make HUC04 col
dswe$HUC04 <- str_pad(substr(as.character(dswe$HUC08), 1, 3), 4, pad = "0")

# make HUC08_SEASON col
dswe$HUC08_SEASON <- paste(as.character(dswe$HUC08), dswe$SEASON, sep="_")


rescov <- function(model, data) {
  var.d <- crossprod(getME(model,"Lambdat"))
  Zt <- getME(model,"Zt")
  vr <- sigma(model)^2
  var.b <- vr*(t(Zt) %*% var.d %*% Zt)
  sI <- vr * Diagonal(nrow(data))
  var.y <- var.b + sI
  invisible(var.y)
}

############################################################################
########                           LMM                              ########
############################################################################

model.huc4.huc8.season.year = lmer(LOG_PR_WATER ~ MAX_TMP + PRECIP + PR_AG + PR_INT + PR_NAT + 
                                    (1|HUC04) + (1|HUC08) + (1|SEASON) + (1|YEAR), 
                                    data = dswe, REML = F)

summary(model.huc4.huc8.season.year)

MuMIn::r.squaredGLMM(model.huc4.huc8.season.year)

step(model.huc4.huc8.season.year)

############################################################################

model.huc4.huc8.season = lmer(LOG_PR_WATER ~ MAX_TMP + PRECIP + PR_AG + PR_INT + PR_NAT + 
                                    (1|HUC04) + (1|HUC08) + (1|SEASON), 
                                    data = dswe, REML = F)

summary(model.huc4.huc8.season)

MuMIn::r.squaredGLMM(model.huc4.huc8.season)

step(model.huc4.huc8.season)


############################################################################

model.huc4.season = lmer(LOG_PR_WATER ~ MAX_TMP + PRECIP + PR_AG + PR_INT + PR_NAT + 
                                    (1|HUC04) + (1|SEASON), 
                                    data = dswe, REML = F)

summary(model.huc4.season)

MuMIn::r.squaredGLMM(model.huc4.season)

step(model.huc4.season)


############################################################################

model.huc8.season = lmer(LOG_PR_WATER ~ MAX_TMP + PRECIP + PR_AG + PR_INT + PR_NAT + 
                                    (1|HUC08) + (1|SEASON), 
                                    data = dswe, REML = F)

summary(model.huc8.season)

MuMIn::r.squaredGLMM(model.huc8.season)

step(model.huc8.season)


############################################################################
####### This is the same as model.huc4.huc8.season !!
####### Which we can do in python because all variables are crossed

model.huc4Xhuc8.season = lmer(LOG_PR_WATER ~ MAX_TMP + PRECIP + PR_AG + PR_INT + PR_NAT + 
                                    (1|HUC04/HUC08) + (1|SEASON), 
                                    data = dswe, REML = F)

summary(model.huc4Xhuc8.season)

MuMIn::r.squaredGLMM(model.huc4Xhuc8.season)

step(model.huc4Xhuc8.season)

# model.huc8Xhuc4.season = lmer(LOG_PR_WATER ~ MAX_TMP + PRECIP + PR_AG + PR_INT + PR_NAT + 
#                                     (1|HUC08/HUC04) + (1|SEASON), 
#                                     data = dswe, REML = F)

# summary(model.huc8Xhuc4.season)

# MuMIn::r.squaredGLMM(model.huc8Xhuc4.season)

# step(model.huc8Xhuc4.season)


model.huc8_season = lmer(LOG_PR_WATER ~ MAX_TMP + PRECIP + PR_AG + PR_INT + PR_NAT + 
                                    (1|HUC08_SEASON), 
                                    data = dswe, REML = F)

summary(model.huc8_season)

MuMIn::r.squaredGLMM(model.huc8_season)

step(model.huc8_season)

# TESTING
dswe_sample = dswe[(dswe$YEAR <= 2003) & ((dswe$HUC08==3010101) | (dswe$HUC08==3100101) |
                    (dswe$HUC08==3100102) | (dswe$HUC08==3010102)),]

dswe_sample <- dswe_sample[order(dswe_sample$HUC04, dswe_sample$HUC08),]

model.huc4Xhuc8.season.sample = lmer(LOG_PR_WATER ~ MAX_TMP + PRECIP + PR_AG + PR_INT + PR_NAT + 
                                    (1|HUC04/HUC08) + (1|SEASON), 
                                    data = dswe_sample, REML = F)

model.huc4.huc8.season.sample = lmer(LOG_PR_WATER ~ MAX_TMP + PRECIP + PR_AG + PR_INT + PR_NAT + 
                                    (1|HUC04) + (1|HUC08) + (1|SEASON), 
                                    data = dswe_sample, REML = F)

rc4b <- rescov(model.huc4Xhuc8.season.sample, dswe_sample)

rc.huc4.huc8.season.sample <- rescov(model.huc4.huc8.season.sample, dswe_sample)

Xi <- getME(model.huc4.huc8.season.sample,"mmList")

Xi_mat <- as.matrix(Xi)

parsedFomula <- lFormula(formula=LOG_PR_WATER ~ MAX_TMP + PRECIP + PR_AG + PR_INT + PR_NAT + 
                                    (1|HUC04) + (1|HUC08) + (1|SEASON), 
                                    data = dswe_sample)

# this is the Z matrix
t(parsedFomula$reTrms$Zt)

(t(parsedFomula$reTrms$Zt))
dim(t(parsedFomula$reTrms$Zt))
dim(dswe_sample)
head(dswe_sample)

model.huc8.sample = lmer(LOG_PR_WATER ~ MAX_TMP + PRECIP + PR_AG + PR_INT + PR_NAT + 
                                    (1|HUC08), 
                                    data = dswe_sample, REML = F)

rc.huc8.sample <- rescov(model.huc8.sample, dswe_sample)

