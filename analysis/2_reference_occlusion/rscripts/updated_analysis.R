library(lme4)
library(dplyr)
library(emmeans)
library(tidyverse)
library(ggplot2)
library(ggsignif)
library(tidytext)
library(RColorBrewer)
library(tibble)
library(purrr)
library(stringr)

theme_set(theme_bw())
# color-blind-friendly palette
cbPalette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7") 
altPalette <- c("#BBBBBB","#CC6677")

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("helpers.R")

# load the files and data
datafile_path <- "../../../data/2_reference_occlusion"
files <- list.files(
  path = datafile_path,
  pattern = "^speaker-.*_annotated\\.csv$",
  full.names = TRUE
)
files <- files[!grepl("_2_annotated.csv", files)]
speaker_data <- map_dfr(files, function(f) {
  file_name <- basename(f)
  parts <- strsplit(file_name, "_")[[1]]
  
  read_csv(f) %>% 
    mutate(model = str_remove(parts[1],"speaker-"),
           reasoning = parts[3])
})

# data_exclusion
exclusions <- tribble(
  ~model, ~reasoning, ~image_file,
  "gemini-2.5-flash", "low", "exp2_044.jpeg",
  "gemini-2.5-pro", "none", "exp2_008.jpeg",
  "gemini-2.5-pro", "none", "exp2_009.jpeg",
  "gemini-2.5-pro", "none", "exp2_013.jpeg",
  "gemini-2.5-pro", "none", "exp2_031.jpeg",
  "gemini-2.5-pro", "none", "exp2_041.jpeg",
  "gemini-2.5-pro", "none", "exp2_065.jpeg",
  "gemini-2.5-pro", "none", "exp2_068.jpeg",
  "gemini-2.5-pro", "none", "exp2_071.jpeg",
  "gemini-2.5-pro", "none", "exp2_103.jpeg"
)
speaker_data_clean <- speaker_data %>% 
  anti_join(exclusions, by = c("model", "reasoning", "image_file"))

# clean up the data
speaker_data_clean <- speaker_data_clean %>% 
  select(-c("speaker_thought", "target_shape_list", "target_texture_list", "speaker_thought_summary")) %>% 
  mutate(occlusion=as.factor(occlusion),
         distractor=as.factor(distractor),
         reasoning = as.factor(reasoning),
         utt_length = str_count(speaker_answer, "\\S+"),
         reasoning = if_else(reasoning == "minimal", "none", reasoning),
         feature_num = rowSums(across((c(contain_shape, contain_color, contain_texture)))))

# average utterance length (overall)
utt_length_summary <- speaker_data_clean %>% 
  group_by(model, reasoning) %>% 
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
ggsave(utt_length_plot,file="../graphs/all_models_speaker_plot.pdf", width=9, height=6)

# plot the average number of features by condition
ggplot(data=feature_summary %>% 
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
ggsave(file="../graphs/all_models_speaker_feature_plot.pdf", width=9, height=6)

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
ggsave(color_feature_plot,file="../graphs/all_models_speaker_color_feature_plot.pdf", width=9, height=6)

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
ggsave(texture_feature_plot,file="../graphs/all_models_speaker_texture_feature_plot.pdf", width=9, height=6)


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
ggsave(shape_feature_plot, file="../graphs/all_models_speaker_shape_feature_plot.pdf", width=9, height=6)


# analysis
# gpt5.1
gpt5.1_data <- speaker_data_clean %>% 
  filter(model == "gpt-5.1")

contrasts(gpt5.1_data$occlusion) <- contr.treatment(2, base = 1)
levels(gpt5.1_data$occlusion) # level1: absent, level2: present
contrasts(gpt5.1_data$distractor) <- contr.treatment(2, base = 1)
levels(gpt5.1_data$distractor) # level1: absent, level2: present
gpt5.1_data$reasoning <- factor(gpt5.1_data$reasoning)
contrasts(gpt5.1_data$reasoning) <- contr.sum(4)
levels(gpt5.1_data$reasoning)

gpt5.1_none_model <- lm(utt_length ~ occlusion * distractor,
                   data=gpt5.1_data %>% 
                     filter(reasoning == "none"))
summary(gpt5.1_none_model)

gpt5.1_low_model <- lm(utt_length ~ occlusion * distractor,
                        data=gpt5.1_data %>% 
                          filter(reasoning == "low"))
summary(gpt5.1_low_model)

gpt5.1_medium_model <- lm(utt_length ~ occlusion * distractor,
                        data=gpt5.1_data %>% 
                          filter(reasoning == "medium"))
summary(gpt5.1_medium_model)

gpt5.1_high_model <- lm(utt_length ~ occlusion * distractor,
                        data=gpt5.1_data %>% 
                          filter(reasoning == "high"))
summary(gpt5.1_high_model)

gpt5.1_all_model <- lm(utt_length ~ occlusion * distractor * reasoning,
                        data=gpt5.1_data)
summary(gpt5.1_all_model)

# gpt5.2
gpt5.2_data <- speaker_data_clean %>% 
  filter(model == "gpt-5.2")

contrasts(gpt5.2_data$occlusion) <- contr.treatment(2, base = 1)
levels(gpt5.2_data$occlusion) # level1: absent, level2: present
contrasts(gpt5.2_data$distractor) <- contr.treatment(2, base = 1)
levels(gpt5.2_data$distractor) # level1: absent, level2: present
gpt5.2_data$reasoning <- factor(gpt5.2_data$reasoning)
contrasts(gpt5.2_data$reasoning) <- contr.sum(4)
levels(gpt5.2_data$reasoning)

gpt5.2_none_model <- lm(utt_length ~ occlusion * distractor,
                        data=gpt5.2_data %>% 
                          filter(reasoning == "none"))
summary(gpt5.2_none_model)

gpt5.2_low_model <- lm(utt_length ~ occlusion * distractor,
                       data=gpt5.2_data %>% 
                         filter(reasoning == "low"))
summary(gpt5.2_low_model)

gpt5.2_medium_model <- lm(utt_length ~ occlusion * distractor,
                          data=gpt5.2_data %>% 
                            filter(reasoning == "medium"))
summary(gpt5.2_medium_model)

gpt5.2_high_model <- lm(utt_length ~ occlusion * distractor,
                        data=gpt5.2_data %>% 
                          filter(reasoning == "high"))
summary(gpt5.2_high_model)

gpt5.2_all_model <- lm(utt_length ~ occlusion * distractor * reasoning,
                       data=gpt5.2_data)
summary(gpt5.2_all_model)

# gemini-2.5-flash
gemini2.5_flash_data <- speaker_data_clean %>% 
  filter(model == "gemini-2.5-flash")

contrasts(gemini2.5_flash_data$occlusion) <- contr.treatment(2, base = 1)
levels(gemini2.5_flash_data$occlusion) # level1: absent, level2: present
contrasts(gemini2.5_flash_data$distractor) <- contr.treatment(2, base = 1)
levels(gemini2.5_flash_data$distractor) # level1: absent, level2: present
gemini2.5_flash_data$reasoning <- factor(gemini2.5_flash_data$reasoning)
contrasts(gemini2.5_flash_data$reasoning) <- contr.sum(4)
levels(gemini2.5_flash_data$reasoning)

gemini2.5_flash_none_model <- lm(utt_length ~ occlusion * distractor,
                        data=gemini2.5_flash_data %>% 
                          filter(reasoning == "none"))
summary(gemini2.5_flash_none_model)

gemini2.5_flash_low_model <- lm(utt_length ~ occlusion * distractor,
                       data=gemini2.5_flash_data %>% 
                         filter(reasoning == "low"))
summary(gemini2.5_flash_low_model)

gemini2.5_flash_medium_model <- lm(utt_length ~ occlusion * distractor,
                          data=gemini2.5_flash_data %>% 
                            filter(reasoning == "medium"))
summary(gemini2.5_flash_medium_model)

gemini2.5_flash_high_model <- lm(utt_length ~ occlusion * distractor,
                        data=gemini2.5_flash_data %>% 
                          filter(reasoning == "high"))
summary(gemini2.5_flash_high_model)

gemini2.5_flash_all_model <- lm(utt_length ~ occlusion * distractor * reasoning,
                              data=gemini2.5_flash_data)
summary(gemini2.5_flash_all_model)

# gemini-2.5-pro
gemini2.5_pro_data <- speaker_data_clean %>% 
  filter(model == "gemini-2.5-pro")

contrasts(gemini2.5_pro_data$occlusion) <- contr.treatment(2, base = 1)
levels(gemini2.5_pro_data$occlusion) # level1: absent, level2: present
contrasts(gemini2.5_pro_data$distractor) <- contr.treatment(2, base = 1)
levels(gemini2.5_pro_data$distractor) # level1: absent, level2: present
gemini2.5_pro_data$reasoning <- factor(gemini2.5_pro_data$reasoning)
contrasts(gemini2.5_pro_data$reasoning) <- contr.sum(4)
levels(gemini2.5_pro_data$reasoning)

gemini2.5_flash_none_model <- lm(utt_length ~ occlusion * distractor,
                                 data=gemini2.5_pro_data %>% 
                                   filter(reasoning == "none"))
summary(gemini2.5_flash_none_model)

gemini2.5_pro_low_model <- lm(utt_length ~ occlusion * distractor,
                                data=gemini2.5_pro_data %>% 
                                  filter(reasoning == "low"))
summary(gemini2.5_pro_low_model)

gemini2.5_pro_medium_model <- lm(utt_length ~ occlusion * distractor,
                                   data=gemini2.5_pro_data %>% 
                                     filter(reasoning == "medium"))
summary(gemini2.5_pro_medium_model)

gemini2.5_pro_high_model <- lm(utt_length ~ occlusion * distractor,
                                 data=gemini2.5_pro_data %>% 
                                   filter(reasoning == "high"))
summary(gemini2.5_pro_high_model)

gemini2.5_pro_all_model <- lm(utt_length ~ occlusion * distractor * reasoning,
                       data=gemini2.5_pro_data)
summary(gemini2.5_pro_all_model)

# gemini-3-flash
gemini3_flash_data <- speaker_data_clean %>% 
  filter(model == "gemini-3-flash")

contrasts(gemini3_flash_data$occlusion) <- contr.treatment(2, base = 1)
levels(gemini3_flash_data$occlusion) # level1: absent, level2: present
contrasts(gemini3_flash_data$distractor) <- contr.treatment(2, base = 1)
levels(gemini3_flash_data$distractor) # level1: absent, level2: present
gemini3_flash_data$reasoning <- factor(gemini3_flash_data$reasoning)
contrasts(gemini3_flash_data$reasoning) <- contr.sum(4)
levels(gemini3_flash_data$reasoning)

gemini3_flash_none_model <- lm(utt_length ~ occlusion * distractor,
                                 data=gemini3_flash_data %>% 
                                   filter(reasoning == "none"))
summary(gemini3_flash_none_model)

gemini3_flash_low_model <- lm(utt_length ~ occlusion * distractor,
                                data=gemini3_flash_data %>% 
                                  filter(reasoning == "low"))
summary(gemini3_flash_low_model)

gemini3_flash_medium_model <- lm(utt_length ~ occlusion * distractor,
                                   data=gemini3_flash_data %>% 
                                     filter(reasoning == "medium"))
summary(gemini3_flash_medium_model)

gemini3_flash_high_model <- lm(utt_length ~ occlusion * distractor,
                                 data=gemini3_flash_data %>% 
                                   filter(reasoning == "high"))
summary(gemini3_flash_high_model)

# gemini-3-pro
gemini3_pro_data <- speaker_data_clean %>% 
  filter(model == "gemini-3-pro")

contrasts(gemini3_pro_data$occlusion) <- contr.treatment(2, base = 1)
levels(gemini3_pro_data$occlusion) # level1: absent, level2: present
contrasts(gemini3_pro_data$distractor) <- contr.treatment(2, base = 1)
levels(gemini3_pro_data$distractor)

gemini3_pro_low_model <- lm(utt_length ~ occlusion * distractor,
                               data=gemini3_pro_data %>% 
                                 filter(reasoning == "low"))
summary(gemini3_pro_low_model)
