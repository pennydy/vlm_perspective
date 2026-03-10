library(lme4)
library(dplyr)
library(emmeans)
library(tidyverse)
library(ggplot2)
library(ggsignif)
library(lmerTest)
library(tidytext)
library(RColorBrewer)
library(tibble)
library(purrr)
library(stringr)
library(broom.mixed)

theme_set(theme_bw())
# color-blind-friendly palette
cbPalette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7") 
altPalette <- c("#BBBBBB","#CC6677")

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("helpers.R")

# 1. Data ----
## 1.1 load the files and data ----
datafile_path <- "../../../data/2_reference_occlusion"
files <- list.files(
  path = datafile_path,
  # pattern = "^speaker-.*_annotated\\.csv$",
  pattern = "^listener-.*_.*_.*_.*\\.csv$",
  full.names = TRUE
)
speaker_data <- map_dfr(files, function(f) {
  file_name <- basename(f)
  parsed <- str_match(
    file_name,
    "^listener-(.+?)_\\d+_([^_]+)_([^_]+)\\.csv$"
  )
  
  read_csv(f) %>%
    mutate(
      model = parsed[2],
      reasoning = parsed[3],
      seed = parsed[4]
    )
})

## 1.2 data exclusion ----
metafile <- read.csv(file.path(datafile_path,"metadata.csv"))
# metafile <- read.csv(file.path(datafile_path,"metadata_include_curtain.csv"))
exclusions <- metafile %>% 
  filter(!is.na(exclusion), exclusion != "") %>% 
  separate_rows(exclusion, sep = ",") %>% 
  mutate(image_file = str_trim(exclusion),
         seed = as.character(seed)) %>%
  select(model, reasoning, seed, image_file)

speaker_data_clean <- speaker_data %>% 
  anti_join(exclusions, by = c("model", "reasoning","seed", "image_file"))

## 1.3 data cleanup ----
speaker_data_clean <- speaker_data_clean %>% 
  select(-c("speaker_thought", "target_shape_list", "target_texture_list", "speaker_thought_summary", "listener_thought_summary")) %>% 
  mutate(occlusion=as.factor(occlusion),
         distractor=as.factor(distractor),
         reasoning = as.factor(reasoning),
         utt_length = str_count(speaker_answer, "\\S+"),
         reasoning = if_else(reasoning == "minimal", "none", reasoning),
         feature_num = rowSums(across((c(contain_shape, contain_color, contain_texture)))))

## 1.4 data summary ----
# average utterance length (overall)
utt_length_summary <- speaker_data_clean %>% 
  group_by(model, reasoning) %>% 
  summarize(mean_utt_length = mean(utt_length))

utt_length_all_summary <- speaker_data_clean %>% 
  group_by(model) %>% 
  summarize(mean_utt_length = mean(utt_length))

# average utterance length (by condition)
utt_length_by_condition_summary <- speaker_data_clean %>% 
  group_by(model, reasoning, occlusion, distractor) %>% 
  summarize(mean_length = mean(utt_length),
            CILow = ci.low(utt_length),
            CIHigh = ci.high(utt_length)) %>% 
  ungroup() %>% 
  mutate(YMin = mean_length-CILow,
         YMax = mean_length+CIHigh)

# average number of features (by condition)
feature_summary <- speaker_data_clean %>% 
  group_by(model, reasoning, occlusion, distractor) %>% 
  summarize(mean_feature = mean(feature_num),
            CILow = ci.low(feature_num),
            CIHigh = ci.high(feature_num)) %>% 
  ungroup() %>% 
  mutate(YMin = mean_feature-CILow,
         YMax = mean_feature+CIHigh)

# number of each features (by condition)
each_feature_summary <- speaker_data_clean %>% 
  mutate(shape=if_else(contain_shape, 1, 0),
         color=if_else(contain_color, 1, 0),
         texture=if_else(contain_texture, 1, 0)) %>% 
  group_by(model, reasoning, occlusion, distractor) %>% 
  summarize(total_count = n(),
            shape=sum(shape)/total_count,
            color=sum(color)/total_count,
            texture=sum(texture)/total_count) %>% 
  ungroup() %>% 
  select(-total_count) %>% 
  pivot_longer(cols=c("shape","color","texture"),
               names_to = "feature_type",
               values_to = "percentage") %>% 
  mutate(feature_type=fct_relevel(feature_type, "shape", "color", "texture"))

# 2. Plots ---- 
## 2.1 utterance length x condition ----
# plot the average utterance length by condition
utt_length_plot <- ggplot(data=utt_length_by_condition_summary %>% 
         mutate(distractor = fct_relevel(distractor, "present", "absent"),
                reasoning = fct_relevel(reasoning, "none", "low", "medium", "high")),
                           aes(x=distractor,y=mean_length,
                               # alpha=distractor,
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
  # scale_alpha_manual(values=c("present"=1,
  #                             "absent"=0.4))+
  facet_grid(reasoning~model)+
  labs(x="distractor",
       y="mean # words")+
  theme(legend.position = "top",
        axis.text = element_text(size=12),
        axis.title = element_text(size=14),
        legend.text = element_text(size=12),
        legend.title = element_text(size=14),
        strip.text.x = element_text(size=12))
utt_length_plot
ggsave(utt_length_plot,file="../graphs/all_models_speaker_with_curtain_plot.pdf", width=9, height=6)

## 2.2 num features x condition ----
# plot the average number of features by condition
feature_plot <- ggplot(data=feature_summary %>% 
         mutate(distractor = fct_relevel(distractor, "present", "absent"),
                reasoning = fct_relevel(reasoning, "none", "low", "medium", "high")),
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
  scale_alpha_manual(values=c("present"=1,
                              "absent"=0.4))+
  facet_grid(reasoning~model)+
  labs(x="distractor",
       y="mean # features")+
  theme(legend.position = "top",
        axis.text = element_text(size=12),
        axis.title = element_text(size=14),
        legend.text = element_text(size=12),
        legend.title = element_text(size=14),
        strip.text.x = element_text(size=12))
feature_plot
ggsave(feature_plot,file="../graphs/all_models_speaker_feature_with_curtain_plot.pdf", width=9, height=6)

## 2.3 each feature x condition ----
# plot percentage of color feature mentioned by condition
color_feature_plot <- ggplot(data=each_feature_summary %>% 
         filter(feature_type == "color") %>% 
         mutate(distractor = fct_relevel(distractor, "present", "absent"),
                reasoning = fct_relevel(reasoning, "none", "low", "medium", "high")),
       aes(x=distractor,y=percentage,fill=occlusion))+
  geom_bar(stat="identity", 
           position=position_dodge(),
           width=0.8, 
           aes(color=occlusion))+
  scale_fill_manual(values=altPalette, name = "occlusion")+
  scale_color_manual(values=altPalette, name = "occlusion")+
  scale_alpha_manual(values=c("present"=1,
                              "absent"=0.4))+
  facet_grid(reasoning~model)+
  labs(x="distractor",
       y="% utterances mentioning color")+
  theme(legend.position = "top",
        axis.text = element_text(size=12),
        axis.title = element_text(size=14),
        legend.text = element_text(size=12),
        legend.title = element_text(size=14),
        strip.text.x = element_text(size=12))
color_feature_plot
ggsave(color_feature_plot,file="../graphs/all_models_speaker_color_feature_with_curtain_plot.pdf", width=9, height=6)

# plot percentage of pattern feature mentioned by condition
texture_feature_plot <- ggplot(data=each_feature_summary %>% 
         filter(feature_type == "texture") %>% 
         mutate(distractor = fct_relevel(distractor, "present", "absent"),
                reasoning = fct_relevel(reasoning, "none", "low", "medium", "high")),
       aes(x=distractor,y=percentage,fill=occlusion))+
  geom_bar(stat="identity", 
           position=position_dodge(),
           width=0.8, 
           aes(color=occlusion))+
  scale_fill_manual(values=altPalette, name = "occlusion")+
  scale_color_manual(values=altPalette, name = "occlusion")+
  scale_alpha_manual(values=c("present"=1,
                              "absent"=0.4))+
  facet_grid(reasoning~model)+
  labs(x="distractor",
       y="% utterances mentioning pattern")+
  theme(legend.position = "top",
        axis.text = element_text(size=12),
        axis.title = element_text(size=14),
        legend.text = element_text(size=12),
        legend.title = element_text(size=14),
        strip.text.x = element_text(size=12))
texture_feature_plot
ggsave(texture_feature_plot,file="../graphs/all_models_speaker_texture_feature_with_curtain_plot.pdf", width=9, height=6)

# plot percentage of shape feature mentioned by condition
shape_feature_plot <- ggplot(data=each_feature_summary %>% 
         filter(feature_type == "shape") %>% 
         mutate(distractor = fct_relevel(distractor, "present", "absent"),
                reasoning = fct_relevel(reasoning, "none", "low", "medium", "high")),
       aes(x=distractor,y=percentage,fill=occlusion))+
  geom_bar(stat="identity", 
           position=position_dodge(),
           width=0.8, 
           aes(color=occlusion))+
  scale_fill_manual(values=altPalette, name = "occlusion")+
  scale_color_manual(values=altPalette, name = "occlusion")+
  scale_alpha_manual(values=c("present"=1,
                              "absent"=0.4))+
  facet_grid(reasoning~model)+
  labs(x="distractor",
       y="% utterances mentioning pattern")+
  theme(legend.position = "top",
        axis.text = element_text(size=12),
        axis.title = element_text(size=14),
        legend.text = element_text(size=12),
        legend.title = element_text(size=14),
        strip.text.x = element_text(size=12))
shape_feature_plot
ggsave(shape_feature_plot, file="../graphs/all_models_speaker_shape_feature_with_curtain_plot.pdf", width=9, height=6)

# 3. Analysis ----
## 3.1 gpt5.1 ----
gpt5.1_data <- speaker_data_clean %>% 
  filter(model == "gpt-5.1")

contrasts(gpt5.1_data$occlusion) <- contr.sum(2)
levels(gpt5.1_data$occlusion) # level1: absent, level2: present
contrasts(gpt5.1_data$distractor) <- contr.sum(2)
levels(gpt5.1_data$distractor) # level1: absent, level2: present
gpt5.1_data$reasoning <- fct_relevel(factor(gpt5.1_data$reasoning),
                                     c("none","low","medium","high"))
contrasts(gpt5.1_data$reasoning) <- contr.sum(4)
levels(gpt5.1_data$reasoning)

gpt5.1_none_model <- lmer(utt_length ~ occlusion * distractor + (1|seed)+(1|image_file) ,
                   data=gpt5.1_data %>% 
                     filter(reasoning == "none"))
summary(gpt5.1_none_model)

gpt5.1_low_model <- lmer(utt_length ~ occlusion * distractor + (1|seed)+(1|image_file),
                        data=gpt5.1_data %>% 
                          filter(reasoning == "low"))
summary(gpt5.1_low_model)

gpt5.1_medium_model <- lmer(utt_length ~ occlusion * distractor + (1|seed)+(1|image_file),
                        data=gpt5.1_data %>% 
                          filter(reasoning == "medium"))
summary(gpt5.1_medium_model)

gpt5.1_high_model <- lmer(utt_length ~ occlusion * distractor + (1|seed)+(1|image_file),
                        data=gpt5.1_data %>% 
                          filter(reasoning == "high"))
summary(gpt5.1_high_model)

# all reasoning levels combined
gpt5.1_all_random_model <- lmer(utt_length ~ occlusion * distractor + (1|seed)+(1|image_file),
                       data=gpt5.1_data)
summary(gpt5.1_all_random_model)

gpt5.1_all_reasoning_model <- lm(utt_length ~ occlusion * distractor * reasoning,
                        data=gpt5.1_data)
summary(gpt5.1_all_reasoning_model)

gpt5.1_all_reasoning_random_model <- lmer(utt_length ~ occlusion * distractor * reasoning + (1|seed) + (1|image_file),
                                 data=gpt5.1_data)
summary(gpt5.1_all_reasoning_random_model)
gpt5.1_all_reasoning_random_anova<-anova(gpt5.1_all_reasoning_random_model, type = 3)
# whether there is an effect of occlusion at each reasoning level
emmeans(gpt5.1_all_reasoning_random_model, pairwise ~ occlusion | reasoning)
# whether there is an effect of distractor at each reasoning level
emmeans(gpt5.1_all_reasoning_random_model, pairwise ~ distractor | reasoning)

gpt5.1_all_reasoning_random_anova <- tidy(gpt5.1_all_reasoning_random_anova)
gpt5.1_all_reasoning_random_anova$model <- "gpt5.1"
gpt5.1_all_reasoning_random_model_result <- tidy(gpt5.1_all_reasoning_random_model)
gpt5.1_all_reasoning_random_model_result$model <- "gpt5.1"

## 3.2 gpt5.2 ----
gpt5.2_data <- speaker_data_clean %>% 
  filter(model == "gpt-5.2")

contrasts(gpt5.2_data$occlusion) <- contr.sum(2)
levels(gpt5.2_data$occlusion) # level1: absent, level2: present
contrasts(gpt5.2_data$distractor) <- contr.sum(2)
levels(gpt5.2_data$distractor) # level1: absent, level2: present
gpt5.2_data$reasoning <- fct_relevel(factor(gpt5.2_data$reasoning),
                                     c("none","low","medium","high"))
contrasts(gpt5.2_data$reasoning) <- contr.sum(4)
levels(gpt5.2_data$reasoning)

gpt5.2_none_model <- lmer(utt_length ~ occlusion * distractor+(1|seed)+(1|image_file),
                        data=gpt5.2_data %>% 
                          filter(reasoning == "none"))
summary(gpt5.2_none_model)

gpt5.2_low_model <- lmer(utt_length ~ occlusion * distractor+(1|seed)+(1|image_file),
                       data=gpt5.2_data %>% 
                         filter(reasoning == "low"))
summary(gpt5.2_low_model)

gpt5.2_medium_model <- lmer(utt_length ~ occlusion * distractor+(1|seed)+(1|image_file),
                          data=gpt5.2_data %>% 
                            filter(reasoning == "medium"))
summary(gpt5.2_medium_model)

gpt5.2_high_model <- lmer(utt_length ~ occlusion * distractor+(1|seed)+(1|image_file),
                        data=gpt5.2_data %>% 
                          filter(reasoning == "high"))
summary(gpt5.2_high_model)

# all reasoning levels combined
gpt5.2_all_random_model <- lmer(utt_length ~ occlusion * distractor + (1|seed)+(1|image_file),
                       data=gpt5.2_data)
summary(gpt5.2_all_random_model)

gpt5.2_all_reasoning_model <- lm(utt_length ~ occlusion * distractor * reasoning,
                       data=gpt5.2_data)
summary(gpt5.2_all_reasoning_model)

gpt5.2_all_reasoning_random_model <- lmer(utt_length ~ occlusion * distractor * reasoning + (1|seed) + (1|image_file),
                                 data=gpt5.2_data)
summary(gpt5.2_all_reasoning_random_model)
gpt5.2_all_reasoning_random_anova<-anova(gpt5.2_all_reasoning_random_model, type = 3)
# whether there is an effect of occlusion at each reasoning level
emmeans(gpt5.2_all_reasoning_random_model, pairwise ~ occlusion | reasoning)
# whether there is an effect of distractor at each reasoning level
emmeans(gpt5.2_all_reasoning_random_model, pairwise ~ distractor | reasoning)

gpt5.2_all_reasoning_random_anova <- tidy(gpt5.2_all_reasoning_random_anova)
gpt5.2_all_reasoning_random_anova$model <- "gpt5.2"
gpt5.2_all_reasoning_random_model_result <- tidy(gpt5.2_all_reasoning_random_model)
gpt5.2_all_reasoning_random_model_result$model <- "gpt5.2"

## 3.3 gemini-2.5-flash ----
gemini2.5_flash_data <- speaker_data_clean %>% 
  filter(model == "gemini-2.5-flash")

contrasts(gemini2.5_flash_data$occlusion) <- contr.sum(2)
levels(gemini2.5_flash_data$occlusion) # level1: absent, level2: present
contrasts(gemini2.5_flash_data$distractor) <- contr.sum(2)
levels(gemini2.5_flash_data$distractor) # level1: absent, level2: present
gemini2.5_flash_data$reasoning <- fct_relevel(factor(gemini2.5_flash_data$reasoning),
                                              c("none","low","medium","high"))
contrasts(gemini2.5_flash_data$reasoning) <- contr.sum(4)
levels(gemini2.5_flash_data$reasoning)

gemini2.5_flash_none_model <- lmer(utt_length ~ occlusion * distractor+(1|seed)+(1|image_file),
                        data=gemini2.5_flash_data %>% 
                          filter(reasoning == "none"))
summary(gemini2.5_flash_none_model)

gemini2.5_flash_low_model <- lmer(utt_length ~ occlusion * distractor+(1|seed)+(1|image_file),
                       data=gemini2.5_flash_data %>% 
                         filter(reasoning == "low"))
summary(gemini2.5_flash_low_model)

gemini2.5_flash_medium_model <- lmer(utt_length ~ occlusion * distractor+(1|seed)+(1|image_file),
                          data=gemini2.5_flash_data %>% 
                            filter(reasoning == "medium"))
summary(gemini2.5_flash_medium_model)

gemini2.5_flash_high_model <- lmer(utt_length ~ occlusion * distractor+(1|seed)+(1|image_file),
                        data=gemini2.5_flash_data %>% 
                          filter(reasoning == "high"))
summary(gemini2.5_flash_high_model)

# all reasoning levels combined
gemini2.5_flash_all_random_model <- lmer(utt_length ~ occlusion * distractor + (1|seed) + (1|image_file),
                                         data=gemini2.5_flash_data)
summary(gemini2.5_flash_all_random_model)

gemini2.5_flash_all_reasoning_model <- lm(utt_length ~ occlusion * distractor * reasoning,
                              data=gemini2.5_flash_data)
summary(gemini2.5_flash_all_reasoning_model)

gemini2.5_flash_all_reasoning_random_model <- lmer(utt_length ~ occlusion * distractor * reasoning + (1|seed) + (1|image_file),
                                          data=gemini2.5_flash_data)
summary(gemini2.5_flash_all_reasoning_random_model)
gemini2.5_flash_all_reasoning_random_anova<-anova(gemini2.5_flash_all_reasoning_random_model, type = 3)
# whether there is an effect of occlusion at each reasoning level
emmeans(gemini2.5_flash_all_reasoning_random_model, pairwise ~ occlusion | reasoning)
# whether there is an effect of distractor at each reasoning level <- this is not justified since the interaction is not significant
emmeans(gemini2.5_flash_all_reasoning_random_model, pairwise ~ distractor | reasoning)

gemini2.5_flash_all_reasoning_random_anova <- tidy(gemini2.5_flash_all_reasoning_random_anova)
gemini2.5_flash_all_reasoning_random_anova$model <- "gemini2.5-flash"
gemini2.5_flash_all_reasoning_random_model_result <- tidy(gemini2.5_flash_all_reasoning_random_model)
gemini2.5_flash_all_reasoning_random_model_result$model <- "gemini2.5-flash"

## 3.4 gemini-2.5-pro ----
gemini2.5_pro_data <- speaker_data_clean %>% 
  filter(model == "gemini-2.5-pro")

contrasts(gemini2.5_pro_data$occlusion) <- contr.sum(2)
levels(gemini2.5_pro_data$occlusion) # level1: absent, level2: present
contrasts(gemini2.5_pro_data$distractor) <- contr.sum(2)
levels(gemini2.5_pro_data$distractor) # level1: absent, level2: present
gemini2.5_pro_data$reasoning <- fct_relevel(factor(gemini2.5_pro_data$reasoning),
                                            c("none","low","medium","high"))
contrasts(gemini2.5_pro_data$reasoning) <- contr.sum(4)
levels(gemini2.5_pro_data$reasoning)

gemini2.5_pro_none_model <- lmer(utt_length ~ occlusion * distractor+(1|seed)+(1|image_file),
                                 data=gemini2.5_pro_data %>% 
                                   filter(reasoning == "none"))
summary(gemini2.5_pro_none_model)

gemini2.5_pro_low_model <- lmer(utt_length ~ occlusion * distractor+(1|seed)+(1|image_file),
                                data=gemini2.5_pro_data %>% 
                                  filter(reasoning == "low"))
summary(gemini2.5_pro_low_model)

gemini2.5_pro_medium_model <- lmer(utt_length ~ occlusion * distractor+(1|seed)+(1|image_file),
                                   data=gemini2.5_pro_data %>% 
                                     filter(reasoning == "medium"))
summary(gemini2.5_pro_medium_model)

gemini2.5_pro_high_model <- lmer(utt_length ~ occlusion * distractor+(1|seed)+(1|image_file),
                                 data=gemini2.5_pro_data %>% 
                                   filter(reasoning == "high"))
summary(gemini2.5_pro_high_model)

# all reasoning levels combined
gemini2.5_pro_all_random_model <- lmer(utt_length ~ occlusion * distractor + (1|seed) + (1|image_file),
                              data=gemini2.5_pro_data)
summary(gemini2.5_pro_all_random_model)

gemini2.5_pro_all_reasoning_model <- lm(utt_length ~ occlusion * distractor * reasoning,
                       data=gemini2.5_pro_data)
summary(gemini2.5_pro_all_reasoning_model)

gemini2.5_pro_all_reasoning_random_model <- lmer(utt_length ~ occlusion * distractor * reasoning + (1|seed) + (1|image_file),
                                        data=gemini2.5_pro_data)
summary(gemini2.5_pro_all_reasoning_random_model)
gemini2.5_pro_all_reasoning_random_anova<-anova(gemini2.5_pro_all_reasoning_random_model, type = 3)
# whether there is an effect of occlusion at each reasoning level
emmeans(gemini2.5_pro_all_reasoning_random_model, pairwise ~ occlusion | reasoning)
# whether there is an effect of distractor at each reasoning level <- this is not justified since the interaction is not significant
emmeans(gemini2.5_pro_all_reasoning_random_model, pairwise ~ distractor | reasoning)

gemini2.5_pro_all_reasoning_random_anova <- tidy(gemini2.5_pro_all_reasoning_random_anova)
gemini2.5_pro_all_reasoning_random_anova$model <- "gemini2.5-pro"
gemini2.5_pro_all_reasoning_random_model_result <- tidy(gemini2.5_pro_all_reasoning_random_model)
gemini2.5_pro_all_reasoning_random_model_result$model <- "gemini2.5-pro"

## 3.5 gemini-3-flash ----
gemini3_flash_data <- speaker_data_clean %>% 
  filter(model == "gemini-3-flash")

contrasts(gemini3_flash_data$occlusion) <- contr.sum(2)
levels(gemini3_flash_data$occlusion) # level1: absent, level2: present
contrasts(gemini3_flash_data$distractor) <- contr.sum(2)
levels(gemini3_flash_data$distractor) # level1: absent, level2: present
gemini3_flash_data$reasoning <- fct_relevel(factor(gemini3_flash_data$reasoning),
                                            c("none","low","medium","high"))
contrasts(gemini3_flash_data$reasoning) <- contr.sum(4)
levels(gemini3_flash_data$reasoning)

gemini3_flash_none_model <- lmer(utt_length ~ occlusion * distractor+(1|seed)+(1|image_file),
                                 data=gemini3_flash_data %>% 
                                   filter(reasoning == "none"))
summary(gemini3_flash_none_model)

gemini3_flash_low_model <- lmer(utt_length ~ occlusion * distractor+(1|seed)+(1|image_file),
                                data=gemini3_flash_data %>% 
                                  filter(reasoning == "low"))
summary(gemini3_flash_low_model)

gemini3_flash_medium_model <- lmer(utt_length ~ occlusion * distractor+(1|seed)+(1|image_file),
                                   data=gemini3_flash_data %>% 
                                     filter(reasoning == "medium"))
summary(gemini3_flash_medium_model)

gemini3_flash_high_model <- lm(utt_length ~ occlusion * distractor,
                                 data=gemini3_flash_data %>% 
                                   filter(reasoning == "high"))
summary(gemini3_flash_high_model)

# all reasoning levels combined
gemini3_flash_all_random_model <- lmer(utt_length ~ occlusion * distractor + (1|seed) + (1|image_file),
                              data=gemini3_flash_data)
summary(gemini3_flash_all_random_model)

gemini3_flash_all_reasoning_model <- lm(utt_length ~ occlusion * distractor * reasoning,
                                          data=gemini3_flash_data)
summary(gemini3_flash_all_reasoning_model)

gemini3_flash_all_reasoning_random_model <- lmer(utt_length ~ occlusion * distractor * reasoning + (1|seed) + (1|image_file),
                                        data=gemini3_flash_data)
summary(gemini3_flash_all_reasoning_random_model)
gemini3_flash_all_reasoning_random_anova<-anova(gemini3_flash_all_reasoning_random_model, type = 3)
# whether there is an effect of occlusion at each reasoning level
emmeans(gemini3_flash_all_reasoning_random_model, pairwise ~ occlusion | reasoning)
# whether there is an effect of distractor at each reasoning level
emmeans(gemini3_flash_all_reasoning_random_model, pairwise ~ distractor | reasoning)

gemini3_flash_all_reasoning_random_anova <- tidy(gemini3_flash_all_reasoning_random_anova)
gemini3_flash_all_reasoning_random_anova$model <- "gemini3-flash"
gemini3_flash_all_reasoning_random_model_result <- tidy(gemini3_flash_all_reasoning_random_model)
gemini3_flash_all_reasoning_random_model_result$model <- "gemini3-flash"

## 3.6 gemini-3-pro ----
gemini3_pro_data <- speaker_data_clean %>% 
  filter(model == "gemini-3-pro")

contrasts(gemini3_pro_data$occlusion) <- contr.sum(2)
levels(gemini3_pro_data$occlusion) # level1: absent, level2: present
contrasts(gemini3_pro_data$distractor) <- contr.sum(2)
levels(gemini3_pro_data$distractor)
gemini3_pro_data$reasoning <- fct_relevel(factor(gemini3_pro_data$reasoning),c("low", "high"))
contrasts(gemini3_pro_data$reasoning) <- contr.sum(2)
levels(gemini3_pro_data$reasoning)

gemini3_pro_low_model <- lmer(utt_length ~ occlusion * distractor+(1|seed)+(1|image_file),
                               data=gemini3_pro_data %>% 
                                 filter(reasoning == "low"))
summary(gemini3_pro_low_model)

gemini3_pro_high_model <- lmer(utt_length ~ occlusion * distractor+(1|seed)+(1|image_file),
                            data=gemini3_pro_data %>% 
                              filter(reasoning == "high"))
summary(gemini3_pro_high_model)

# all reasoning levels combined
gemini3_pro_all_random_model <- lmer(utt_length ~ occlusion * distractor + (1|seed) + (1|image_file),
                            data=gemini3_pro_data)
summary(gemini3_pro_all_random_model)

gemini3_pro_all_reasoning_model <- lm(utt_length ~ occlusion * distractor * reasoning,
                                          data=gemini3_pro_data)
summary(gemini3_pro_all_reasoning_model)

gemini3_pro_all_reasoning_random_model <- lmer(utt_length ~ occlusion * distractor * reasoning + (1|seed) + (1|image_file),
                                      data=gemini3_pro_data)
summary(gemini3_pro_all_reasoning_random_model)
gemini3_pro_all_reasoning_random_anova<-anova(gemini3_pro_all_reasoning_random_model, type = 3) # in theory not needed
# whether there is an effect of occlusion at each reasoning level <- this is not justified since the interaction is not significant
emmeans(gemini3_pro_all_reasoning_random_model, pairwise ~ occlusion | reasoning)
# whether there is an effect of distractor at each reasoning level <- this is not justified since the interaction is not significant
emmeans(gemini3_pro_all_reasoning_random_model, pairwise ~ distractor | reasoning)

gemini3_pro_all_reasoning_random_anova <- tidy(gemini3_pro_all_reasoning_random_anova)
gemini3_pro_all_reasoning_random_anova$model <- "gemini3-pro"
gemini3_pro_all_reasoning_random_model_result <- tidy(gemini3_pro_all_reasoning_random_model)
gemini3_pro_all_reasoning_random_model_result$model <- "gemini3-pro"

## 3.7 all results ----
lmer_results <- bind_rows(
  gpt5.1_all_reasoning_random_model_result,
  gpt5.2_all_reasoning_random_model_result,
  gemini2.5_flash_all_reasoning_random_model_result,
  gemini2.5_pro_all_reasoning_random_model_result,
  gemini3_flash_all_reasoning_random_model_result,
  gemini3_pro_all_reasoning_random_model_result) %>% 
  relocate(model)
write.csv(lmer_results, "../stats_summaries/lmer_results_clean.csv", row.names = FALSE)
# write.csv(lmer_results, "../stats_summaries/lmer_results_include_curtain_clean.csv", row.names = FALSE)

anova_results <- bind_rows(
  gpt5.1_all_reasoning_random_anova,
  gpt5.2_all_reasoning_random_anova,
  gemini2.5_flash_all_reasoning_random_anova,
  gemini2.5_pro_all_reasoning_random_anova,
  gemini3_flash_all_reasoning_random_anova,
  gemini3_pro_all_reasoning_random_anova
) %>% 
  relocate(model)
# write.csv(anova_results, "../stats_summaries/anova_results_clean.csv", row.names = FALSE)
write.csv(anova_results, "../stats_summaries/anova_results_include_curtain_clean.csv", row.names = FALSE)


# 4. Others ----
# replicating the original gemini2.5 pro results
gemini2.5_pro_medium_0 <- read.csv("../../../data/2_reference_occlusion/temperature0/listener-gemini-2.5-pro_0_medium_1.csv",header=TRUE)
gemini2.5_pro_medium_0 <- gemini2.5_pro_medium_0 %>% 
  mutate(occlusion=as.factor(occlusion),
         distractor=as.factor(distractor),
         utt_length = str_count(speaker_answer, "\\S+"))
mean(gemini2.5_pro_medium_0$utt_length) # 9.06
contrasts(gemini2.5_pro_medium_0$occlusion) <- contr.treatment(2, base = 1)
levels(gemini2.5_pro_medium_0$occlusion) # level1: absent, level2: present
contrasts(gemini2.5_pro_medium_0$distractor) <- contr.treatment(2, base = 1)
levels(gemini2.5_pro_medium_0$distractor)
gemini2.5_pro_medium_0_model <- lm(utt_length ~ occlusion * distractor,
                            data=gemini2.5_pro_medium_0)
summary(gemini2.5_pro_medium_0_model)
