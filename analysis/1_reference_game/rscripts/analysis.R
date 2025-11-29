library(lme4)
library(dplyr)
library(emmeans)
library(tidyverse)
library(ggplot2)
library(ggsignif)
library(tidytext)
library(RColorBrewer)
library(stringr)

theme_set(theme_bw())
# color-blind-friendly palette
cbPalette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7") 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("helpers.R")

# 1. Data ----
rsa_summary <- read.csv("../../../data/1_reference_game/rsa_predictions.csv", header=TRUE)
gpt41_speaker.data <- read.csv("../../../data/1_reference_game/speaker_gpt-4.1_1.csv", header=TRUE) %>% 
  na.omit()

gpt41_summary <- gpt41_speaker.data %>% 
  filter(answer1!="none") %>% 
  mutate(answer1 = as.numeric(answer1),
         answer2 = as.numeric(answer2)) %>% 
  group_by(condition) %>% 
  summarize(mean_answer1 = mean(answer1),
            mean_answer2 = mean(answer2), # answer1 is 1-answer2
            CILow = ci.low(answer2),
            CIHigh = ci.high(answer2)) %>% 
  ungroup() %>% 
  mutate(YMin = mean_answer2-CILow,
         YMax = mean_answer2+CIHigh)

rsa_gpt41_summary <- merge(rsa_summary, gpt41_summary, by="condition") %>% 
  rename("RSA" = feature2,
         "gpt4.1" = mean_answer2) %>% 
  select(-c("feature1", "mean_answer1"))

ggplot(data=rsa_gpt41_summary,
       aes(x=RSA,y=gpt4.1))+
  geom_point()+
  geom_abline(intercept = 0, slope = 1, color = "grey", linetype = "dashed") +
  geom_errorbar(aes(ymin=YMin,ymax=YMax), width=.2, show.legend = FALSE) +
  geom_smooth(stat="smooth",method = lm,se=FALSE,color="red") # +
  # scale_x_continuous(limits = c(50, 100)) +
  # scale_y_continuous(limits = c(50, 100))
