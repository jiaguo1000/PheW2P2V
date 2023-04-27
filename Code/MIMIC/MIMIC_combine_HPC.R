library(tidyverse)

res = NULL
for (i in 1:772) {
  filename = paste0("../../Output/MIMIC_task_", i, ".csv")
  tmp = read_csv(filename)
  res = rbind(res, tmp)
}
out_name = paste0("../../Output/MIMIC_res.csv")
write_csv(res, out_name)