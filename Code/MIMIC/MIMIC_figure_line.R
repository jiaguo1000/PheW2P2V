library(tidyverse)
source("../R_ggplot.R")
res = read_csv("../../Output/MIMIC_res.csv")
disease = read_csv("Data/MIMIC_outcome_count.csv") %>% select(-group)

G = list()
m_list = c("auc", "f1", "ap")
y_list = c("AUC-ROC", "F1-score", "AUC-PR")
k = 300
for (gi in 1:length(m_list)) {
  metric = m_list[gi]
  res = rawdata %>% 
    group_by(phecode) %>% 
    mutate(LR = mean(!!sym(paste0("LR_", metric))),
           RF = mean(!!sym(paste0("RF_", metric))),
           XGB = mean(!!sym(paste0("XGB_", metric))),
           P2V = mean(!!sym(paste0("P2V_", metric))),
           WP2V = mean(!!sym(paste0("WP2V_", metric)))) %>% 
    select(phecode, LR, RF, XGB, P2V, WP2V) %>% 
    ungroup() %>% 
    distinct()
  
  n = nrow(res)
  identical(res$phecode, disease$phecode[1:n])
  
  m_Q1_Q3 = function(m, Q1, Q3, digits=3){
    m = formatC(m, digits, format = "f")
    Q1 = formatC(Q1, digits, format = "f")
    Q3 = formatC(Q3, digits, format = "f")
    return( paste0(m," (", Q1, ", ", Q3, ")") )
  }
  
  # group -------------------------------------------------------------------
  output = res %>% left_join(disease)
  output = rbind(output, output)
  output$group = NA
  
  n = nrow(res)
  cut_list = floor(n/k)
  cut_list = (0:cut_list)*k
  cut_list = c(cut_list, n)
  if (n-cut_list[length(cut_list)-1]<50) {cut_list = cut_list[-(length(cut_list)-1)]}
  for (i in 1:(length(cut_list)-1)) {
    lhs = cut_list[i]+1
    rhs = cut_list[i+1]
    tmp = paste0(lhs, "_", rhs)
    output = output %>% 
      mutate(group = ifelse(row_number()>=lhs & row_number()<=rhs, tmp, group))
  }
  output = output %>% 
    mutate(group = ifelse(row_number()>=(n+1), "all", group))
  
  # table -------------------------------------------------------------------
  l = 0.25
  h = 0.75
  output = output %>% 
    group_by(group) %>% 
    mutate(m_case_p = median(case_p),
           Q1_case_p = quantile(case_p, l),
           Q3_case_p = quantile(case_p, h),
           
           m_LR = median(LR),
           Q1_LR = quantile(LR, l),
           Q3_LR = quantile(LR, h),
           
           m_RF = median(RF),
           Q1_RF = quantile(RF, l),
           Q3_RF = quantile(RF, h),
           
           m_XGB = median(XGB),
           Q1_XGB = quantile(XGB, l),
           Q3_XGB = quantile(XGB, h),
           
           m_PV = median(P2V),
           Q1_PV = quantile(P2V, l),
           Q3_PV = quantile(P2V, h),
           
           m_WPV = median(WP2V),
           Q1_WPV = quantile(WP2V, l),
           Q3_WPV = quantile(WP2V, h),
           
           n_WPV_PV = sum(WP2V>P2V),
           n_WPV_LR = sum(WP2V>LR),
           n_WPV_RF = sum(WP2V>RF),
           n_WPV_XGB = sum(WP2V>XGB)
    ) %>% 
    ungroup() %>% 
    select(group, m_case_p, Q1_case_p, Q3_case_p,
           m_LR, Q1_LR, Q3_LR,
           m_RF, Q1_RF, Q3_RF,
           m_XGB, Q1_XGB, Q3_XGB,
           m_PV, Q1_PV, Q3_PV,
           m_WPV,Q1_WPV, Q3_WPV,
           n_WPV_PV, n_WPV_LR, n_WPV_RF, n_WPV_XGB) %>% 
    distinct()
  
  # plot --------------------------------------------------------------------
  output = output[output$group!="all",]
  plot_data = tibble(rank = rep(1:length(output$m_LR), 5),
                     F1 = c(output$m_LR, output$m_RF, output$m_XGB, output$m_PV, output$m_WPV),
                     lower = c(output$Q1_LR, output$Q1_RF, output$Q1_XGB, output$Q1_PV, output$Q1_WPV),
                     upper = c(output$Q3_LR, output$Q3_RF, output$Q3_XGB, output$Q3_PV, output$Q3_WPV),
                     method = rep(c("regression", "random forest", "gradient boosted tree", "P2V", "PheWP2V"),
                                  each=length(output$m_LR)))
  
  x_lab = formatC(output$m_case_p, 3, format = "f")
  x_lab = paste0("Rank: ", str_replace(output$group, "_", "-"),
                 "\n median p=", x_lab)
  
  plot_data$method = factor(plot_data$method,
                            levels = c("PheWP2V", "P2V", "regression", "random forest", "gradient boosted tree"))
  
  temp_theme = my_theme
  temp_theme$axis.title.x = element_blank()
  temp_theme$axis.text.x$size = 15
  temp_theme$legend.position = "none"
  G[[gi]] = ggplot(data = plot_data, aes(x=rank, y=F1))+
    # geom_line(aes(color=method), size=1.5)+
    geom_line(aes(linetype=method), size=1.4)+
    geom_point(size=1.5)+
    geom_errorbar(aes(ymin=lower, ymax=upper), size=0.8, width=0.08, alpha=0.6)+
    scale_x_continuous(breaks = 1:length(output$m_LR), labels = x_lab)+
    # scale_y_continuous(limits = c(0.45, 0.9))+
    scale_color_manual(values = my_color(4))+
    scale_linetype_manual(values = c("solid", "dashed", "dotdash", "dotted", "longdash"))+
    labs(y = y_list[gi])+
    temp_theme+
    theme(axis.text.x=element_text(angle=45, hjust=1))
}
main_plot = plot_grid(plotlist = G, ncol = 3)

temp_theme = my_theme
temp_theme$legend.position = "top"
temp_theme$legend.key.width = unit(5, "lines")
legend_plot = get_legend(ggplot(data = plot_data, aes(x=rank, y=F1))+
                           # geom_line(aes(color=method), size=1.5)+
                           geom_line(aes(linetype=method), size=1.4)+
                           geom_point(size=1.5)+
                           geom_errorbar(aes(ymin=lower, ymax=upper), size=0.8, width=0.08, alpha=0.6)+
                           scale_x_continuous(breaks = 1:length(output$m_LR), labels = x_lab)+
                           scale_linetype_manual(values = c("solid", "dashed", "dotdash", "dotted", "longdash"),
                                                 labels = c(expression(PheW^2*P2V), "P2V", "regression", "random forest", "gradient boosted tree"))+
                           temp_theme)

G_title = ggdraw() + 
  draw_label(paste0("Prediction performance of different methods on 933 phenotypes in the MIMIC-III database"),
             fontface = "bold", size = 22)

G_out = plot_grid(G_title, legend_plot, main_plot,
                  ncol = 1, rel_heights = c(0.10, 0.12, 1))

ggsave(paste0("../../Output/mimic_figure.png"),
       plot = G_out,
       width = 6*3, height = 7*1, dpi = 300)






