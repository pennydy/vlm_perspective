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
altPalette <- c("#BBBBBB","#CC6677")

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("helpers.R")

# 1. Data ----
gpt4.1_speaker.data <- read.csv("../../../data/2_reference_occlusion/speaker-gpt-4.1_2.csv", header=TRUE)

gpt4.1_speaker_summary <- gpt4.1_speaker.data %>% 
  mutate(utt_length = str_count(speaker_answer, "\\S+")) %>% 
  group_by(occlusion, distractor) %>% 
  summarize(mean_length = mean(utt_length),
            CILow = ci.low(utt_length),
            CIHigh = ci.high(utt_length)) %>% 
  ungroup() %>% 
  mutate(YMin = mean_length-CILow,
         YMax = mean_length+CIHigh)

gpt41_speaker_plot <- ggplot(data=gpt4.1_speaker_summary,
                             aes(x=distractor,y=mean_length,fill=occlusion))+
  geom_bar(stat="identity", 
           position=position_dodge(),
           width=0.8, 
           aes(color=occlusion))+
  geom_errorbar(aes(ymin=YMin,
                    ymax=YMax),
                width=.2,
                position=position_dodge(width=0.8),
                show.legend = FALSE) +
  scale_fill_manual(values=altPalette, name = "occlusion")+
  scale_color_manual(values=altPalette, name = "occlusion")+
  labs(x="distractor",
       y="mean # words")+
  theme(legend.position = "top")
gpt41_speaker_plot
ggsave(gpt41_speaker_plot, file="../graphs/gpt41_speaker_plot.pdf", width=4, height=3)
