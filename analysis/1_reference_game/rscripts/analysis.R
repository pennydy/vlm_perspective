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
gpt4.1_speaker.data <- read.csv("../../../data/1_reference_game/speaker_gpt-4.1_1.csv", header=TRUE) %>% 
  na.omit()
gpt4.1_free_speaker.data <- read.csv("../../../data/1_reference_game/free_speaker_gpt4.1_3.csv", header=TRUE) %>% 
  filter(target_feature_mentioned=="yes")

gpt5.1_free_speaker.data <- read.csv("../../../data/1_reference_game/free_speaker_gpt5.1_3.csv", header=TRUE)

gpt5.1_low_free_speaker.data <- read.csv("../../../data/1_reference_game/free_speaker_gpt5.1_low_3.csv", header=TRUE)

gemini2.5_speaker.data <- read.csv("../../../data/1_reference_game/speaker_gemini-2.5-pro_1.csv", header=TRUE) %>% 
  na.omit()

gemini2.5_free_speaker.data <- read.csv("../../../data/1_reference_game/free_speaker_gemini2.5-pro_3.csv", header=TRUE) %>% 
  filter(target_feature_mentioned=="yes")

# 2. Summary ----
## 2.1. GPT-4.1 ----
gpt4.1_summary <- gpt4.1_speaker.data %>% 
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

rsa_gpt4.1_summary <- merge(rsa_summary, gpt4.1_summary, by="condition") %>% 
  rename("RSA" = feature2,
         "gpt4.1" = mean_answer2) %>% 
  select(-c("feature1", "mean_answer1"))

ggplot(data=rsa_gpt4.1_summary,
       aes(x=RSA,y=gpt4.1))+
  geom_point()+
  geom_abline(intercept = 0, slope = 1, color = "grey", linetype = "dashed") +
  geom_errorbar(aes(ymin=YMin,ymax=YMax), width=.2, show.legend = FALSE) +
  geom_smooth(stat="smooth",method = lm,se=FALSE,color="red") # +
  # scale_x_continuous(limits = c(50, 100)) +
  # scale_y_continuous(limits = c(50, 100))

sum(gpt4.1_free_speaker.data$target_feature_mentioned=="no") # 0
sum(gpt4.1_free_speaker.data$target_feature_mentioned=="none") # 1
gpt4.1_free_speaker_summary <- gpt4.1_free_speaker.data %>% 
  filter(feature!="none") %>% 
  filter(target_feature_mentioned=="yes") %>% 
  mutate(feature=as.integer(feature)) %>% 
  group_by(condition) %>% 
  summarize(mean_feature = mean(feature),
            CILow = ci.low(feature),
            CIHigh = ci.high(feature)) %>% 
  ungroup() %>% 
  mutate(YMin = mean_feature-CILow,
         YMax = mean_feature+CIHigh)

## 2.2. GPT-5.1 ----
sum(gpt5.1_free_speaker.data$target_feature_mentioned=="no") # 0
sum(gpt5.1_free_speaker.data$target_feature_mentioned=="none") # 0
gpt5.1_free_speaker_summary <- gpt5.1_free_speaker.data %>% 
  filter(target_feature_mentioned=="yes") %>% 
  group_by(condition) %>% 
  summarize(mean_feature = mean(feature),
            CILow = ci.low(feature),
            CIHigh = ci.high(feature)) %>% 
  ungroup() %>% 
  mutate(YMin = mean_feature-CILow,
         YMax = mean_feature+CIHigh)

## 2.2.1 GPT-5.1 ----
sum(gpt5.1_low_free_speaker.data$target_feature_mentioned=="no") # 0
sum(gpt5.1_low_free_speaker.data$target_feature_mentioned=="none") # 0
gpt5.1_low_free_speaker_summary <- gpt5.1_low_free_speaker.data %>% 
  filter(target_feature_mentioned=="yes") %>% 
  group_by(condition) %>% 
  summarize(mean_feature = mean(feature),
            CILow = ci.low(feature),
            CIHigh = ci.high(feature)) %>% 
  ungroup() %>% 
  mutate(YMin = mean_feature-CILow,
         YMax = mean_feature+CIHigh)

## 2.3. Gemini-2.5 ----
gemini2.5_summary <- gemini2.5_speaker.data %>% 
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

rsa_gemini25_summary <- merge(rsa_summary, gemini2.5_summary, by="condition") %>% 
  rename("RSA" = feature2,
         "gemine2.5" = mean_answer2) %>% 
  select(-c("feature1", "mean_answer1"))

ggplot(data=rsa_gemini25_summary,
       aes(x=RSA,y=gemine2.5))+
  geom_point()+
  geom_abline(intercept = 0, slope = 1, color = "grey", linetype = "dashed") +
  geom_errorbar(aes(ymin=YMin,ymax=YMax), width=.2, show.legend = FALSE) +
  geom_smooth(stat="smooth",method = lm,se=FALSE,color="red")

sum(gemini2.5_free_speaker.data$target_feature_mentioned=="no") # 22
sum(gemini2.5_free_speaker.data$target_feature_mentioned=="none") # 0
gemini2.5_free_speaker_summary <- gemini2.5_free_speaker.data %>% 
  filter(feature!="none") %>% 
  filter(target_feature_mentioned=="yes") %>% 
  mutate(feature=as.integer(feature)) %>% 
  group_by(condition) %>% 
  summarize(mean_feature = mean(feature),
            CILow = ci.low(feature),
            CIHigh = ci.high(feature)) %>% 
  ungroup() %>% 
  mutate(YMin = mean_feature-CILow,
         YMax = mean_feature+CIHigh)

## 2.4. all models ----
gpt4.1_free_speaker_summary$model = "GPT-4.1"
gpt5.1_free_speaker_summary$model = "GPT-5.1"
gpt5.1_low_free_speaker_summary$model = "GPT-5.1 low"
gemini2.5_free_speaker_summary$model = "Gemini-2.5-pro"
all_models_summary <- bind_rows(gpt4.1_free_speaker_summary,gpt5.1_free_speaker_summary, gpt5.1_low_free_speaker_summary,gemini2.5_free_speaker_summary)

# 3. Plot ----
all_free_speaker_plot <- ggplot(data=all_models_summary %>%
         mutate(condition=case_when(condition=="cond3" ~ "one-\ntwo",
                                    condition=="cond4_diff"~"zero",
                                    condition=="cond5" ~ "one-\none",
                                    condition=="cond6" ~ "two")),
       aes(x=condition,y=mean_feature,fill=condition))+
  geom_bar(stat="identity", 
           position=position_dodge(),
           width=0.8)+
  geom_errorbar(aes(ymin=YMin,
                    ymax=YMax),
                width=.2,
                position=position_dodge(width=0.8),
                show.legend = FALSE) +
  scale_fill_manual(values=cbPalette, guide = "none")+
  scale_color_manual(values=cbPalette, guide = "none")+
  facet_grid(.~model)+
  labs(x="condition",
       y="mean # features")+
  theme(legend.position = "top",
        axis.text = element_text(size=10),
        axis.title = element_text(size=14),
        legend.text = element_text(size=12),
        legend.title = element_text(size=14),
        strip.text.x = element_text(size=12)) +  
  scale_y_continuous(limits = c(0, 3))
all_free_speaker_plot
ggsave(all_free_speaker_plot, file="../graphs/all_models_free_speaker_plot_filtered.pdf", width=7, height=3)

# 4. Analysis ----
# cond3-> one-two, cond5-> one-one, cond6->two
# 4.1. GPT-4.1 ----
gpt4.1_free_speaker.data$condition <- as.factor(gpt4.1_free_speaker.data$condition)
contrasts(gpt4.1_free_speaker.data$condition) <- contr.treatment(4, base=2)
levels(gpt4.1_free_speaker.data$condition)

gpt4.1_model <- lm(feature ~ condition, data=gpt4.1_free_speaker.data)
summary(gpt4.1_model)

# 4.2. GPT-5.1 ----
gpt5.1_free_speaker.data$condition <- as.factor(gpt5.1_free_speaker.data$condition)
contrasts(gpt5.1_free_speaker.data$condition) <- contr.treatment(4, base=2)
levels(gpt5.1_free_speaker.data$condition)

gpt5.1_model <- lm(feature ~ condition, data=gpt5.1_speaker.data)
summary(gpt5.1_model)

# 4.2.1. GPT-5.1 low ----
gpt5.1_low_free_speaker.data$condition <- as.factor(gpt5.1_low_free_speaker.data$condition)
contrasts(gpt5.1_low_free_speaker.data$condition) <- contr.treatment(4, base=2)
levels(gpt5.1_low_free_speaker.data$condition)

gpt5.1_low_model <- lm(feature ~ condition,
                       data=gpt5.1_low_free_speaker.data)
summary(gpt5.1_low_model)

# 4.3. Gemini-2.5 ----
gemini2.5_free_speaker.data$condition <- as.factor(gemini2.5_free_speaker.data$condition)
contrasts(gemini2.5_free_speaker.data$condition) <- contr.treatment(4, base=2)
levels(gemini2.5_free_speaker.data$condition)

gemini2.5_model <- lm(feature ~ condition, data=gemini2.5_free_speaker.data)
summary(gemini2.5_model)
