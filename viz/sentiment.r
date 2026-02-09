# ============================================================
#  Congress Sentiment Engine: Full Batch Version
#  - Prevents "Long Vector" errors via 10k batching
#  - Custom Algorithm (Dictionary + Negation)
#  - Auto-column detection to prevent 'closure' errors
# ============================================================

.libPaths("~/R/x86_64-pc-linux-gnu-library/4.4")

suppressPackageStartupMessages({
  library(data.table)
  library(stringr)
  library(lubridate)
  library(tidytext)
  library(ggplot2)
  library(scales)
})

# -------------------------
# 0) Configuration
# -------------------------
# MANDATORY: Ensure these paths are correct and in quotes
data_path <- 
output_dir <- 
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

lexicon_url <- "https://raw.githubusercontent.com/cjhutto/vaderSentiment/master/vaderSentiment/vader_lexicon.txt"
BATCH_SIZE  <- 10000 

# -------------------------
# 1) Robust Column Loading
# -------------------------
message("Step 1: Loading data and mapping columns...")

# Check header first
header <- names(fread(data_path, nrows = 0))

# Helper function to find columns
find_col <- function(pats, choices) {
  # Exact match (case insensitive)
  match <- choices[tolower(choices) %in% pats]
  # Partial match
  if (length(match) == 0) {
    pattern_str <- paste(pats, collapse = "|")
    match <- grep(pattern_str, choices, ignore.case = TRUE, value = TRUE)
  }
  return(if(length(match) > 0) match[1] else NA)
}

col_id    <- find_col(c("speech_id", "id", "doc_id", "record_id"), header)
col_text  <- find_col(c("text", "speech_text", "body", "content", "raw_text"), header)
col_date  <- find_col(c("date", "speech_date", "datetime"), header)
col_speak <- find_col(c("speaker", "name", "orator", "member"), header)

# Verify we have the essentials
if (is.na(col_text) || is.na(col_date)) {
  stop("Could not find Text or Date columns. Check your CSV header.")
}

# Read only necessary columns
dt <- fread(data_path, select = c(col_id, col_text, col_date, col_speak))
setnames(dt, c(col_id, col_text, col_date, col_speak), c("speech_id", "text", "date", "speaker"))

# Basic Cleaning
dt[, date := lubridate::as_date(date)]
dt[, year := year(date)]
dt <- dt[!is.na(text) & nchar(text) > 5 & !is.na(year)]
dt[, row_idx := .I] # Essential for batch tracking

message("  - Total speeches to process: ", format(nrow(dt), big.mark=","))

# -------------------------
# 2) Sentiment Lexicon Setup
# -------------------------
message("Step 2: Preparing Sentiment Lexicon...")
lex <- fread(lexicon_url, select = c(1, 2))
setnames(lex, c("word", "score"))
setkey(lex, word) # Fast lookup key

# Words that flip sentiment
negators <- c("not", "no", "never", "neither", "nor", "without", "lack", "none", "cannot", "dont", "didnt")

# -------------------------
# 3) Batch Sentiment Processing
# -------------------------
message("Step 3: Processing in batches of ", BATCH_SIZE, "...")

n_speeches <- nrow(dt)
n_batches  <- ceiling(n_speeches / BATCH_SIZE)
results_list <- vector("list", n_batches)

for (i in 1:n_batches) {
  start_idx <- ((i - 1) * BATCH_SIZE) + 1
  end_idx   <- min(i * BATCH_SIZE, n_speeches)
  
  # A. Subset batch
  batch_dt <- dt[start_idx:end_idx, .(row_idx, speech_id, text)]
  
  # B. Tokenize (Avoids Long Vector error by staying under 2.1B rows)
  tidy_batch <- batch_dt %>%
    tidytext::unnest_tokens(word, text) %>%
    as.data.table()
  
  # C. Negation Handling
  # Look at previous word within the same speech index
  tidy_batch[, prev_word := shift(word, 1, type = "lag"), by = row_idx]
  
  # D. Score Matching
  scored <- lex[tidy_batch, on = "word", nomatch = NULL]
  
  # Flip score if preceded by a negator (dampen by 0.8 for nuance)
  scored[prev_word %in% negators, score := score * -0.8]
  
  # E. Aggregation (Sum and Count)
  batch_results <- scored[, .(
    sent_total = sum(score),
    sent_words = .N
  ), by = .(row_idx, speech_id)]
  
  results_list[[i]] <- batch_results
  
  if (i %% 10 == 0 || i == n_batches) {
    message("  - Progress: Batch ", i, " of ", n_batches, " (", end_idx, " speeches)")
  }
}

# -------------------------
# 4) Final Merge and Normalization
# -------------------------
message("Step 4: Consolidating scores...")

all_sentiment <- rbindlist(results_list)

# Compound score normalization: -1 to 1
# Formula: x / sqrt(x^2 + alpha), where alpha=15
all_sentiment[, sentiment_compound := sent_total / sqrt(sent_total^2 + 15)]

# Merge back to original data
final_dt <- merge(dt, all_sentiment[, .(row_idx, sentiment_compound, sent_words)], 
                  by = "row_idx", all.x = TRUE)

# Fill gaps for speeches with zero matches
final_dt[is.na(sentiment_compound), `:=`(sentiment_compound = 0, sent_words = 0)]

# -------------------------
# 5) Outputs and Visualization
# -------------------------
message("Step 5: Exporting files and plotting...")

# Full CSV Output
fwrite(final_dt[, .(speech_id, date, year, speaker, sentiment_compound, sent_words)], 
       file.path(output_dir, "sentiment_full_results.csv"))

# Yearly Aggregation
yearly_stats <- final_dt[, .(
  mean_sentiment = mean(sentiment_compound),
  n_speeches = .N
), by = year][order(year)]

fwrite(yearly_stats, file.path(output_dir, "sentiment_by_year.csv"))

# The Plot
p <- ggplot(yearly_stats, aes(x = year, y = mean_sentiment)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  geom_line(color = "#2c3e50", linewidth = 1) +
  geom_smooth(method = "loess", color = "#e74c3c", fill = "#e74c3c", alpha = 0.1) +
  scale_y_continuous(labels = number_format(accuracy = 0.01)) +
  theme_minimal(base_size = 14) +
  labs(
    title = "U.S. Congress Sentiment Trend",
    subtitle = "Calculated via Batch Lexicon Algorithm with Negation Logic",
    x = "Year",
    y = "Average Sentiment Score (-1 to 1)",
    caption = paste("Processed", format(n_speeches, big.mark=","), "speeches")
  )

ggsave(file.path(output_dir, "sentiment_trend.png"), p, width = 12, height = 6, dpi = 300)

message("DONE! Results are in: ", output_dir)
