library(curatedMetagenomicData)
library(curatedMetagenomicAnalyses)
library(tidyverse)
library(mia)

T2D_studies = c(
  "MetaCardis_2020_a", # large
  "KarlssonFH_2013", # small
  "QinJ_2012" # small
)
T2D = sampleMetadata %>%
  filter(study_name %in% T2D_studies) %>%
  select(study_name, sample_id, subject_id, body_site, antibiotics_current_use, study_condition,
         disease, age, gender, country, BMI)
T2Dmeta = T2D %>% filter(
  disease=="healthy" | str_detect(disease, "T2D")
) %>% mutate(
  T2D = ifelse(str_detect(disease, "T2D"), 1, 0)
)

# get relative abundance
T2Dse = T2Dmeta %>%
  returnSamples("relative_abundance", rownames = "short")
T2Dser = agglomerateByRank(T2Dse, "genus")
T2Drel = assay(T2Dser) %>% data.frame() %>% t()

# save to CSV
write.csv(T2Dmeta, "t2d_meta.csv", row.names = FALSE)
write.csv(T2Drel, "t2d_rel.csv", row.names = TRUE)


# summary statistics

table1::table1(~ age + gender + BMI + disease + country + as.factor(T2D) + antibiotics_current_use| study_name, data=T2Dmeta)
