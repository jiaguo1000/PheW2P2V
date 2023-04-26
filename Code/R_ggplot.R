# library(conflicted)
library(ggplot2)
library(viridis)
library(scales)
library(cowplot)
library(RColorBrewer)
library(ggtext)

MacOS_path = "~/OneDrive/MyWorkMyLife/"
Windows_path = "C:/Users/Jia/OneDrive/MyWorkMyLife/"

my_color = function(n){
  tmp = c("#1F78B4", "#B0315BFF", "#61564A", "#78B41F", "#FF8B3D",
          "#8B3DFF", "#00CCCC", "#A7B8B2", "#FDE725FF", "white")
  return(tmp[1:n])
}

show_col(my_color(10))
dev.off()

my_theme = theme_classic()+
  theme(axis.title=element_text(size=22),
        axis.text=element_text(size=20),
        # legend.title=element_text(size=18),
        legend.title=element_blank(),
        legend.text=element_text(size=20),
        panel.grid.major.y = element_line(color = "grey", linetype = "dashed", linewidth = 0.6),
        # legend.position = c(0.90, 0.90),
        # axis.text.x=element_text(angle=45, hjust=1),
        plot.title = element_text(size=22, hjust = 0.5, face = "bold"))


