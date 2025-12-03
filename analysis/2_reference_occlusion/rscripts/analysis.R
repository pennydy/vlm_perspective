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
gpt4.1_speaker.data <- read.csv("../../../data/2_reference_occlusion/listener-gpt-4.1_2.csv", header=TRUE) %>% 
  filter(listener_answer_correct == "yes")
length(gpt4.1_speaker.data$image_file)/120 # 0.925

gpt5.1_speaker.data <- read.csv("../../../data/2_reference_occlusion/listener-gpt-5.1_2.csv", header=TRUE) %>% 
  filter(listener_answer_correct == "yes")
length(gpt5.1_speaker.data$image_file)/120 # 0.99

gemini25_speaker.data <- read.csv("../../../data/2_reference_occlusion/listener-gemini2.5_0.csv", header=TRUE)  %>% 
  filter(listener_answer_correct == "yes")
length(gemini25_speaker.data$image_file)/120 # 0.94

qwen3_speaker.data <- read.csv("../../../data/2_reference_occlusion/speaker-qwen_1.csv", header=TRUE) # the listener results are very bad...
qwen3_speaker.data <- qwen3_speaker.data %>% 
  filter(! image_file %in% c("exp2_064.jpeg", "exp2_071.jpeg", "exp2_101.jpeg", "exp2_107.jpeg")) # excluding cases where it uses absolute positions and like the "highlighted box"


# 2. Summary ----
## 2.1. GPT-4.1 ----
gpt4.1_speaker_summary <- gpt4.1_speaker.data %>% 
  mutate(utt_length = str_count(speaker_answer, "\\S+")) %>% 
  group_by(occlusion, distractor) %>% 
  summarize(mean_length = mean(utt_length),
            CILow = ci.low(utt_length),
            CIHigh = ci.high(utt_length)) %>% 
  ungroup() %>% 
  mutate(YMin = mean_length-CILow,
         YMax = mean_length+CIHigh)

gpt4.1_speaker_feature_summary <- gpt4.1_speaker.data %>% 
  group_by(occlusion, distractor) %>% 
  summarize(mean_feature = mean(feature),
            CILow = ci.low(feature),
            CIHigh = ci.high(feature)) %>% 
  ungroup() %>% 
  mutate(YMin = mean_feature-CILow,
         YMax = mean_feature+CIHigh)

gpt4.1_speaker_each_feature_summary <- gpt4.1_speaker.data %>% 
  mutate(shape=if_else(shape=="yes", 1, 0),
         color=if_else(color=="yes", 1, 0),
         texture=if_else(texture=="yes", 1, 0)) %>% 
  group_by(occlusion, distractor) %>% 
  summarize(shape=sum(shape)/30,
            color=sum(color)/30,
            texture=sum(texture)/30) %>% 
  ungroup() %>% 
  pivot_longer(cols=c("shape","color","texture"),
               names_to = "feature_type",
               values_to = "percentage") %>% 
  mutate(feature_type=fct_relevel(feature_type, "shape", "color", "texture"))

## 2.2. GPT-5.1 ----
gpt5.1_speaker_summary <- gpt5.1_speaker.data %>% 
  mutate(utt_length = str_count(speaker_answer, "\\S+")) %>% 
  group_by(occlusion, distractor) %>% 
  summarize(mean_length = mean(utt_length),
            CILow = ci.low(utt_length),
            CIHigh = ci.high(utt_length)) %>% 
  ungroup() %>% 
  mutate(YMin = mean_length-CILow,
         YMax = mean_length+CIHigh)

## 2.3. Gemini-2.5 ----
gemini25_speaker_summary <- gemini25_speaker.data %>% 
  mutate(utt_length = str_count(speaker_answer, "\\S+")) %>% 
  group_by(occlusion, distractor) %>% 
  summarize(mean_length = mean(utt_length),
            CILow = ci.low(utt_length),
            CIHigh = ci.high(utt_length)) %>% 
  ungroup() %>% 
  mutate(YMin = mean_length-CILow,
         YMax = mean_length+CIHigh)

gemini25_speaker_feature_summary <- gemini25_speaker.data %>% 
  group_by(occlusion, distractor) %>% 
  summarize(mean_feature = mean(feature),
            CILow = ci.low(feature),
            CIHigh = ci.high(feature)) %>% 
  ungroup() %>% 
  mutate(YMin = mean_feature-CILow,
         YMax = mean_feature+CIHigh)

gemini25_speaker_each_feature_summary <- gemini25_speaker.data %>% 
  mutate(shape=if_else(shape=="yes", 1, 0),
         color=if_else(color=="yes", 1, 0),
         texture=if_else(texture=="yes", 1, 0)) %>% 
  group_by(occlusion, distractor) %>% 
  summarize(shape=sum(shape)/30,
            color=sum(color)/30,
            texture=sum(texture)/30) %>% 
  ungroup() %>% 
  pivot_longer(cols=c("shape","color","texture"),
               names_to = "feature_type",
               values_to = "percentage") %>% 
  mutate(feature_type=fct_relevel(feature_type, "shape", "color", "texture"))

## 2.4. Qwen3-VL ----
qwen3_speaker_summary <- qwen3_speaker.data %>% 
  mutate(utt_length = str_count(speaker_answer, "\\S+")) %>% 
  group_by(occlusion, distractor) %>% 
  summarize(mean_length = mean(utt_length),
            CILow = ci.low(utt_length),
            CIHigh = ci.high(utt_length)) %>% 
  ungroup() %>% 
  mutate(YMin = mean_length-CILow,
         YMax = mean_length+CIHigh)

qwen3_speaker_feature_summary <- qwen3_speaker.data %>% 
  group_by(occlusion, distractor) %>% 
  summarize(mean_feature = mean(feature),
            CILow = ci.low(feature),
            CIHigh = ci.high(feature)) %>% 
  ungroup() %>% 
  mutate(YMin = mean_feature-CILow,
         YMax = mean_feature+CIHigh)

qwen3_speaker_each_feature_summary <- qwen3_speaker.data %>% 
  mutate(shape=if_else(shape=="yes", 1, 0),
         color=if_else(color=="yes", 1, 0),
         texture=if_else(texture=="yes", 1, 0)) %>% 
  group_by(occlusion, distractor) %>% 
  summarize(shape=sum(shape)/30, # two cond1s and two cond4s are excluded
            color=sum(color)/30,
            texture=sum(texture)/30) %>% 
  ungroup() %>% 
  pivot_longer(cols=c("shape","color","texture"),
               names_to = "feature_type",
               values_to = "percentage") %>% 
  mutate(feature_type=fct_relevel(feature_type, "shape", "color", "texture"))

gpt4.1_speaker_summary$model = "GPT-4.1"
gpt5.1_speaker_summary$model = "GPT-5.1"
gemini25_speaker_summary$model = "Gemini-2.5"
qwen3_speaker_summary$model = "Qwen3-VL-8B-Instruct"
all_models_summary <- bind_rows(gpt4.1_speaker_summary,gpt5.1_speaker_summary, gemini25_speaker_summary,qwen3_speaker_summary)

# 3. Plot ----
## 3.1. GPT-4.1 ----
gpt41_speaker_plot <- ggplot(data=gpt4.1_speaker_summary %>% 
                               mutate(distractor = fct_relevel(distractor, "present", "absent")),
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
  theme(legend.position = "top",
        axis.text = element_text(size=12),
        axis.title = element_text(size=14),
        legend.text = element_text(size=12),
        legend.title = element_text(size=14))
gpt41_speaker_plot
ggsave(gpt41_speaker_plot, file="../graphs/gpt41_speaker_plot.pdf", width=4, height=3)

ggplot(data=gpt4.1_speaker_feature_summary %>% 
         mutate(distractor = fct_relevel(distractor, "present", "absent")),,
       aes(x=distractor,y=mean_feature,fill=occlusion))+
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
       y="mean # features")+
  theme(legend.position = "top")

ggplot(data=gpt4.1_speaker_each_feature_summary %>% 
         mutate(distractor = fct_relevel(distractor, "present", "absent")),
       aes(x=distractor,y=percentage,fill=occlusion))+
  geom_bar(stat="identity", 
           position=position_dodge(),
           width=0.8, 
           aes(color=occlusion))+
  scale_fill_manual(values=altPalette, name = "occlusion")+
  scale_color_manual(values=altPalette, name = "occlusion")+
  facet_grid(.~feature_type)+
  labs(x="distractor",
       y="mean # features")+
  theme(legend.position = "top")

## 3.2. GPT-5.1 ----
gpt51_speaker_plot <- ggplot(data=gpt5.1_speaker_summary %>% 
                               mutate(distractor = fct_relevel(distractor, "present", "absent")),
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
  theme(legend.position = "top",
        axis.text = element_text(size=12),
        axis.title = element_text(size=14),
        legend.text = element_text(size=12),
        legend.title = element_text(size=14))
gpt51_speaker_plot
ggsave(gpt51_speaker_plot, file="../graphs/gpt51_speaker_plot.pdf", width=4, height=3)

## 3.3. Gemini-2.5 ----
gemini25_speaker_plot <- ggplot(data=gemini25_speaker_summary %>% 
                                  mutate(distractor = fct_relevel(distractor, "present", "absent")),
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
  theme(legend.position = "top",
        axis.text = element_text(size=12),
        axis.title = element_text(size=14),
        legend.text = element_text(size=12),
        legend.title = element_text(size=14))
gemini25_speaker_plot
ggsave(gemini25_speaker_plot, file="../graphs/gemini25_speaker_plot.pdf", width=4, height=3)

ggplot(data=gemini25_speaker_feature_summary,
       aes(x=distractor,y=mean_feature,fill=occlusion))+
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
       y="mean # features")+
  theme(legend.position = "top")

ggplot(data=gemini25_speaker_each_feature_summary %>% 
         mutate(distractor = fct_relevel(distractor, "present", "absent")),
       aes(x=distractor,y=percentage,fill=occlusion))+
  geom_bar(stat="identity", 
           position=position_dodge(),
           width=0.8, 
           aes(color=occlusion))+
  scale_fill_manual(values=altPalette, name = "occlusion")+
  scale_color_manual(values=altPalette, name = "occlusion")+
  facet_grid(.~feature_type)+
  labs(x="distractor",
       y="mean # features")+
  theme(legend.position = "top")

## 3.4. Qwen3-VL ----
qwen3_speaker_plot <- ggplot(data=qwen3_speaker_summary %>% 
                               mutate(distractor = fct_relevel(distractor, "present", "absent")),
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
  theme(legend.position = "top",
        axis.text = element_text(size=12),
        axis.title = element_text(size=14),
        legend.text = element_text(size=12),
        legend.title = element_text(size=14))
qwen3_speaker_plot
ggsave(qwen3_speaker_plot, file="../graphs/qwen3_speaker_plot.pdf", width=4, height=3)

ggplot(data=qwen3_speaker_feature_summary,
       aes(x=distractor,y=mean_feature,fill=occlusion))+
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
       y="mean # features")+
  theme(legend.position = "top")

all_speaker_plot <- ggplot(data=all_models_summary %>% 
                             mutate(distractor = fct_relevel(distractor, "present", "absent")),
                                    # alpha_value = ifelse(distractor == "absent", 0, 1)), 
                           aes(x=distractor,y=mean_length,
                               # alpha=alpha_value,
                               fill=occlusion))+
  geom_bar(stat="identity", 
           position=position_dodge(),
           width=0.8)+
  geom_errorbar(aes(ymin=YMin,
                    ymax=YMax),
                width=.2,
                position=position_dodge(width=0.8),
                show.legend = FALSE) +
  scale_fill_manual(values=altPalette, name = "occlusion")+
  scale_color_manual(values=altPalette, name = "occlusion")+
  # guides(alpha="none")+
  facet_wrap(.~model)+
  labs(x="distractor",
       y="mean # words")+
  theme(legend.position = "top",
        axis.text = element_text(size=12),
        axis.title = element_text(size=14),
        legend.text = element_text(size=12),
        legend.title = element_text(size=14),
        strip.text.x = element_text(size=12))
all_speaker_plot
ggsave(all_speaker_plot, file="../graphs/all_models_speaker_plot.pdf", width=6, height=5)
