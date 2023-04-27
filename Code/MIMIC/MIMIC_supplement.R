library(tidyverse)
rawdata = read_csv("../../Output/MIMIC_res.csv")
disease = read_csv("Data/MIMIC_outcome_count.csv")

# function ----------------------------------------------------------------
metric = "auc"
res_1 = rawdata %>% 
  group_by(phecode) %>% 
  summarise(!!paste0("LR_", "AUCROC") := mean(!!sym(paste0("LR_", metric))),
            !!paste0("RF_", "AUCROC") := mean(!!sym(paste0("RF_", metric))),
            !!paste0("XGB_", "AUCROC") := mean(!!sym(paste0("XGB_", metric))),
            !!paste0("P2V_", "AUCROC") := mean(!!sym(paste0("P2V_", metric))),
            !!paste0("WP2V_", "AUCROC") := mean(!!sym(paste0("WP2V_", metric))))

metric = "f1"
res_2 = rawdata %>% 
  group_by(phecode) %>% 
  summarise(!!paste0("LR_", "F1") := mean(!!sym(paste0("LR_", metric))),
            !!paste0("RF_", "F1") := mean(!!sym(paste0("RF_", metric))),
            !!paste0("XGB_", "F1") := mean(!!sym(paste0("XGB_", metric))),
            !!paste0("P2V_", "F1") := mean(!!sym(paste0("P2V_", metric))),
            !!paste0("WP2V_", "F1") := mean(!!sym(paste0("WP2V_", metric))))

metric = "ap"
res_3 = rawdata %>% 
  group_by(phecode) %>% 
  summarise(!!paste0("LR_", "AUCPR") := mean(!!sym(paste0("LR_", metric))),
            !!paste0("RF_", "AUCPR") := mean(!!sym(paste0("RF_", metric))),
            !!paste0("XGB_", "AUCPR") := mean(!!sym(paste0("XGB_", metric))),
            !!paste0("P2V_", "AUCPR") := mean(!!sym(paste0("P2V_", metric))),
            !!paste0("WP2V_", "AUCPR") := mean(!!sym(paste0("WP2V_", metric))))

res_0 = rawdata %>% 
  select(phecode, rank) %>% 
  distinct()

output = disease %>% 
  left_join(res_0) %>% 
  left_join(res_1) %>% 
  left_join(res_2) %>% 
  left_join(res_3) %>% 
  select(phecode, rank, description, group, cases:case_p, everything())

write_csv(output, paste0("../../Output/Supplement_B.csv"))
