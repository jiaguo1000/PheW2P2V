# output phenotype info ---------------------------------------------------
library(PheWAS)
library(tidyverse)

code_map = PheWAS::phecode_map %>% filter(vocabulary_id=="ICD9CM")
info = PheWAS::pheinfo

write_csv(code_map, "Data/phecode_ICD9.csv")
write_csv(info, "Data/phecode_info.csv")

# phecode for all adm -----------------------------------------------------
library(PheWAS)
library(tidyverse)
rawdata = read_csv("Data/all_diag_ICD9.csv")

work_data = rawdata %>% 
  rename(id = HADM_ID, code = ICD9, sex = GENDER) %>% 
  mutate(vocabulary_id = "ICD9CM", count = 1) %>% 
  select(id, vocabulary_id, code, count, sex)

id_sex = work_data %>% select(id, sex) %>% distinct()
phe = createPhenotypes(work_data %>% select(id:count), min.code.count = 1, id.sex = id_sex)

idx = c(1, colSums(phe[,2:ncol(phe)], na.rm = T))
res = phe[,idx>0]

output = NULL
t0 = proc.time()
for (i in 1:nrow(res)) {
  tmp = res %>% select(-id) %>% slice(i) %>% as_vector() %>% na.omit()
  tmp = tibble(HADM_ID = res$id[i], phecode = names(tmp)[tmp])
  output = bind_rows(output, tmp)
  if (i%%1000==0) {message(i, " - ", (proc.time()-t0)[3])}
}

saveRDS(output, "Data/all_diag_phecode.rds")

# get case-control from the last visit ------------------------------------
library(PheWAS)
library(tidyverse)
rawdata = read_csv("Data/all_diag_ICD9.csv")
outcome_id = read_csv("Data/outcome_HADM.csv")

work_data = rawdata %>% 
  rename(id = HADM_ID, code = ICD9, sex = GENDER) %>% 
  mutate(vocabulary_id = "ICD9CM", count = 1) %>% 
  select(id, vocabulary_id, code, count, sex)

id_sex = work_data %>% select(id, sex) %>% distinct()
phe = createPhenotypes(work_data %>% select(id:count), min.code.count = 1, id.sex = id_sex)

idx = c(1, colSums(phe[,2:ncol(phe)], na.rm = T))
res = phe[,idx>0]

outcome = outcome_id %>% 
  rename(id = outcome_HADM) %>% 
  left_join(res) %>% 
  select(-id)

cases = NULL
ctrls = NULL
t0 = proc.time()
for (i in 1:nrow(outcome)) {
  tmp = outcome %>% select(-SUBJECT_ID) %>% slice(i) %>% as_vector() %>% na.omit()
  tmp1 = tibble(SUBJECT_ID = outcome$SUBJECT_ID[i], phecode = names(tmp)[tmp])
  tmp2 = tibble(SUBJECT_ID = outcome$SUBJECT_ID[i], phecode = names(tmp)[!tmp])
  cases = bind_rows(cases, tmp1)
  ctrls = bind_rows(ctrls, tmp2)
  if (i%%500==0) {message(i, " - ", (proc.time()-t0)[3])}
}

saveRDS(cases, "Data/outcome_cases.rds")
saveRDS(ctrls, "Data/outcome_ctrls.rds")
