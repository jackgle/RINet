# download_liver_dataset_updated_v4.r

# ---- install/load libraries -------------------------------------------------
if (!requireNamespace("reflimR", quietly = TRUE)) {
  install.packages("reflimR", repos = "https://cloud.r-project.org/")
}
library(reflimR)

# ---- download raw data ------------------------------------------------------
write.csv(livertests, "reflimr_liver.csv", row.names = FALSE)

download.file(
  "https://archive.ics.uci.edu/static/public/571/hcv+data.zip",
  destfile = "uci_liver.zip",
  method   = "auto"
)
unzip("uci_liver.zip", exdir = ".")

# ---- load datasets ----------------------------------------------------------
df1 <- read.csv("./hcvdat0.csv", stringsAsFactors = FALSE)
df1 <- df1[ , -1]
df2 <- read.csv("./reflimr_liver.csv", stringsAsFactors = FALSE)

## NEW: capture UCI-only columns to add later (row-aligned)
uci_only_cols <- setdiff(colnames(df1), colnames(df2))
df1_extra <- if (length(uci_only_cols)) df1[, uci_only_cols, drop = FALSE] else NULL

# ---- harmonize datasets -----------------------------------------------------
# drop ALP/CHOL if present
drop_cols <- intersect(c("ALP", "CHOL"), colnames(df1))
if (length(drop_cols) > 0) df1 <- df1[ , !(colnames(df1) %in% drop_cols), drop = FALSE]

# keep the same row subset in the extra columns too
keep_idx <- complete.cases(df1)          # (equivalent to na.omit, but lets us mirror the mask)
df1 <- df1[keep_idx, , drop = FALSE]
if (!is.null(df1_extra)) df1_extra <- df1_extra[keep_idx, , drop = FALSE]

# sort both by measurements (all except Category)
sort_cols <- setdiff(colnames(df1), "Category")

ord1 <- do.call(order, df1[sort_cols])
df1 <- df1[ ord1, ]
if (!is.null(df1_extra)) df1_extra <- df1_extra[ ord1, , drop = FALSE ]

ord2 <- do.call(order, df2[sort_cols])
df2 <- df2[ ord2, ]

# check identical measurements
if (!all(df1[ , sort_cols] == df2[ , sort_cols])) {
  stop("Error: dataset measurements are not identical")
}

## NEW: append UCI-only columns to df2 (now rows are aligned)
if (!is.null(df1_extra)) {
  for (cn in colnames(df1_extra)) df2[[cn]] <- df1_extra[[cn]]
}

# ---- reconcile labels -------------------------------------------------------
is_uci_reference <- grepl("^0", df1$Category)
df2$Category[!is_uci_reference] <- df1$Category[!is_uci_reference]
df <- df2

# relabel categories (unchanged)
label_dict <- c(
  "reference"              = "reference",
  "patient"                = "abnormal",
  "0=Blood Donor"          = "reference",
  "0s=suspect Blood Donor" = "reference",
  "1=Hepatitis"            = "hepatitis",
  "2=Fibrosis"             = "fibrosis",
  "3=Cirrhosis"            = "cirrhosis"
)
df$Category <- unname(label_dict[df$Category])

# ---- format columns ---------------------------------------------------------
df$label  <- df$Category
df$gender <- toupper(df$Sex)
df$age    <- df$Age

# drop unused columns but KEEP CHE now
for (col in c("Category", "Sex", "Age")) {
  if (col %in% colnames(df)) df[[col]] <- NULL
}

# ---- rename analytes --------------------------------------------------------
analyte_dict <- c(
  "ALB"  = "albumin",
  "ALP"  = "alkaline phosphatase",
  "ALT"  = "alanine aminotransferase",
  "AST"  = "aspartate aminotransferase",
  "BIL"  = "bilirubin",
  "CHOL" = "cholesterol",
  "CREA" = "creatinine",
  "GGT"  = "gamma-glutamyl transferase",
  "PROT" = "total protein",
  "CHE"  = "cholinesterase"
)
colnames(df) <- ifelse(colnames(df) %in% names(analyte_dict),
                       analyte_dict[colnames(df)],
                       colnames(df))

# ---- unit conversions -------------------------------------------------------
if ("albumin" %in% colnames(df))        df$albumin        <- df$albumin / 10
if ("total protein" %in% colnames(df))  df$`total protein`<- df$`total protein` / 10
if ("creatinine" %in% colnames(df))     df$creatinine     <- df$creatinine * 0.01131
if ("bilirubin" %in% colnames(df))      df$bilirubin      <- df$bilirubin * 0.05847
if ("cholesterol" %in% colnames(df))    df$cholesterol    <- df$cholesterol * 0.02586
if ("alkaline phosphatase" %in% colnames(df)) df$`alkaline phosphatase` <- df$`alkaline phosphatase` * 1
if ("cholinesterase" %in% colnames(df)) df$cholinesterase <- df$cholinesterase * 1

# ---- finalize dataset -------------------------------------------------------
first_cols <- c("gender", "age", "label")
other_cols <- setdiff(colnames(df), first_cols)
df <- df[ , c(first_cols, sort(other_cols))]

# save
write.csv(df, "./liver_preprocessed.csv", row.names = FALSE)

# ---- summary ----------------------------------------------------------------
cat("Summary:\n\n")
cat("No. records:", nrow(df), "\n")
cat("Path. frac.:", mean(df$label != "reference"), "\n")
cat("Label counts:")
print(table(df$label))
cat("\n")
