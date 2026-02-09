# ============================================================
#  Filibuster Visualization – Clean, Fixed, Fully Working Code
# ============================================================

.libPaths("~/R/x86_64-pc-linux-gnu-library/4.4")

# Libraries
library(ggplot2)
library(dplyr)
library(tidyr)
library(stringr)
library(lubridate)
library(glue)
library(jsonlite)
library(httr)
library(scales)
library(purrr)
library(ggrepel)

message("Step 0: Libraries loaded.")

# -------------------------
# 0) Config
# -------------------------
data_path <- 

top_n_per_decade <- 4
k_words <- 5

output_dir <- 
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

out_all   <- file.path(output_dir, "dotplot_all_speeches.png")
out_tfidf <- file.path(output_dir, "dotplot_labeled_tfidf.png")
out_groq  <- file.path(output_dir, "dotplot_labeled_groq.png")
out_top_csv <- file.path(output_dir, "top_speeches_per_decade.csv")

groq_model <- "llama-3.3-70b-versatile"
groq_url   <- "https://api.groq.com/openai/v1/chat/completions"
Sys.setenv(GROQ_API_KEY = )

# -------------------------
# 1) Load data
# -------------------------
message("Step 1: Loading data from CSV...")

raw <- read.csv(data_path, stringsAsFactors = FALSE)

pick_col <- function(df, candidates) {
  hits <- intersect(names(df), candidates)
  if (length(hits) == 0) return(NA_character_)
  hits[1]
}

col_id    <- pick_col(raw, c("speech_id","id","doc_id","record_id"))
col_speak <- pick_col(raw, c("speaker","orator","name","member","person","speaker_name"))
col_date  <- pick_col(raw, c("date","speech_date","datetime","time","timestamp"))
col_text  <- pick_col(raw, c("text","speech_text","body","content","transcript","raw_text"))
col_wc    <- pick_col(raw, c("word_count","words","n_words"))

needed <- c(col_speak, col_date, col_text)
if (any(is.na(needed))) stop("Missing required columns.")

speeches <- raw %>%
  mutate(
    .speech_id = if (!is.na(col_id)) .data[[col_id]] else row_number(),
    .speaker = .data[[col_speak]],
    .date = as_date(.data[[col_date]]),
    .text = .data[[col_text]],
    .word_count = if (!is.na(col_wc)) as.integer(.data[[col_wc]])
                  else str_count(.data[[col_text]], boundary("word"))
  ) %>%
  transmute(
    speech_id = .speech_id,
    speaker = as.character(.speaker),
    date = .date,
    text = as.character(.text),
    word_count = .word_count
  ) %>%
  mutate(
    year = year(date),
    decade = (year %/% 10) * 10
  ) %>%
  filter(!is.na(date), !is.na(word_count),
         word_count > 0,
         year >= 1880, year <= 2025)

message("  - Number of speeches after filtering: ", nrow(speeches))

# -------------------------
# 2) Base theme
# -------------------------
message("Step 2: Creating all-speeches dot plot...")

base_theme <- theme_minimal(base_size = 14) +
  theme(
    plot.background  = element_rect(fill = "black", color = NA),
    panel.background = element_rect(fill = "black", color = NA),
    panel.grid.major.x = element_line(color = "#333333", linewidth = 0.4),
    panel.grid.major.y = element_line(color = "#333333", linewidth = 0.4),

    axis.text.x = element_text(color = "white", size = 10,
                               angle = 90, vjust = 0.5, hjust = 1),
    axis.text.y = element_text(color = "white", size = 10),
    axis.title  = element_text(color = "white", size = 12),

    plot.title    = element_text(color = "white", size = 18, face = "bold"),
    plot.subtitle = element_text(color = "white", size = 12),
    plot.caption  = element_text(color = "white", size = 10)
  )

# -------------------------
# Plot all speeches
# -------------------------
p_all <- ggplot(speeches, aes(x = date, y = word_count)) +
  geom_point(
    alpha = 0.04, size = 0.4, color = "white",
    position = position_jitter(width = 0.3, height = 0)
  ) +
  scale_y_log10(labels = comma_format()) +
  scale_x_date(date_breaks = "5 years", date_labels = "%Y") +
  labs(
    title = "Speech Lengths Over Time",
    subtitle = "Each dot is a speech; y-axis is log(word count)",
    x = "Date",
    y = "Word Count (log10)"
  ) +
  base_theme

ggsave(out_all, p_all, width = 13, height = 7.5, dpi = 600)
message("  - Saved plot: ", out_all)

# -------------------------
# 3) Top speeches per decade
# -------------------------
message("Step 3: Finding top speeches per decade...")

top_by_decade <- speeches %>%
  group_by(decade) %>%
  slice_max(order_by = word_count, n = top_n_per_decade, with_ties = FALSE) %>%
  ungroup()

message("  - Number of top speeches: ", nrow(top_by_decade))

# -------------------------
# 4) TF-IDF subjects
# -------------------------
message("Step 4: Computing TF–IDF subjects (only for top speeches)...")

stop_words_vec <- c(
  "a","an","the","and","or","but","if","then","than","so",
  "of","in","on","at","to","for","from","by","with",
  "is","are","was","were","be","been","being",
  "i","you","he","she","it","we","they","them","their",
  "this","that","these","those","there","here",
  "because","about","into","over","under","up","down"
)

top_speeches <- speeches %>%
  semi_join(top_by_decade, by = "speech_id")

tidy_top <- top_speeches %>%
  select(speech_id, text) %>%
  mutate(text = tolower(text),
         text = str_replace_all(text, "[^a-z' ]", " "),
         text = str_squish(text)) %>%
  separate_rows(text, sep = "\\s+") %>%
  rename(word = text) %>%
  filter(word != "",
         !word %in% stop_words_vec,
         str_detect(word, "^[a-z][a-z'-]*$"))

word_counts <- tidy_top %>% count(speech_id, word)
totals <- word_counts %>% group_by(speech_id) %>% summarise(total = sum(n))
docfreq <- word_counts %>% group_by(word) %>% summarise(df = n_distinct(speech_id))
N <- n_distinct(word_counts$speech_id)

tfidf <- word_counts %>%
  left_join(totals, by = "speech_id") %>%
  left_join(docfreq, by = "word") %>%
  mutate(tf = n / total,
         idf = log(N / df),
         tfidf = tf * idf)

subjects_tfidf <- tfidf %>%
  group_by(speech_id) %>%
  slice_max(order_by = tfidf, n = k_words, with_ties = FALSE) %>%
  summarise(subject_tfidf = str_to_sentence(paste(word, collapse = " ")))

top_by_decade_tfidf <- top_by_decade %>%
  left_join(subjects_tfidf, by = "speech_id") %>%
  mutate(label_tfidf = glue("{speaker}\n{date}\n{subject_tfidf}"))

# -------------------------
# 5) TF-IDF plot (FIXED)
# -------------------------
message("Step 5: Creating TF-IDF labeled plot...")

p_tfidf <- ggplot(speeches, aes(x = date, y = word_count)) +
  geom_point(
    aes(size = sqrt(word_count)),
    alpha = 0.08, color = "white",
    position = position_jitter(width = 0.3)
  ) +
  scale_size(range = c(1.2, 6), guide = "none") +
  scale_y_log10(labels = comma) +
  scale_x_date(date_breaks = "5 years", date_labels = "%Y") +

  geom_point(
    data = top_by_decade_tfidf,
    aes(size = sqrt(word_count)),
    color = "white"
  ) +

  geom_label_repel(
    data = top_by_decade_tfidf,
    aes(label = label_tfidf),
    fill = "white",
    color = "black",
    size = 2.5,
    box.padding = 1,
    point.padding = 0.5,
    force = 3,
    max.overlaps = Inf,
    min.segment.length = 0,
    segment.color = "white",        
    segment.curvature = -0.2,
    segment.angle = 15,
    segment.ncp = 8
  ) +

  labs(
    title = "Longest Speeches by Decade (TF-IDF Subjects)",
    subtitle = "Top speeches per decade",
    x = "Date",
    y = "Word Count (log10)"
  ) +
  base_theme

ggsave(out_tfidf, p_tfidf, width = 28, height = 18, dpi = 600)
message("  - Saved plot: ", out_tfidf)

# -------------------------
# 6) Groq subjects
# -------------------------
message("Step 6: Querying Groq for subject labels...")

truncate_for_llm <- function(x, max_chars=3000) {
  if (is.na(x) || !nzchar(trimws(x))) return("")
  if (nchar(x) > max_chars) paste0(substr(x, 1, max_chars), "…") else x
}

get_groq_subject <- function(text) {
  # Always return length-1 character
  if (is.na(text) || !nzchar(trimws(text))) return("EMPTY_TEXT")

  body <- list(
    model = groq_model,
    messages = list(
      list(role = "system",
           content = "Return exactly ONE topic label of 3–5 words. No quotes, no punctuation, no extra text."),
      list(role = "user",
           content = paste("Label this speech in 3–5 words:", truncate_for_llm(text)))
    ),
    temperature = 0.2,
    max_tokens = 12
  )

  resp <- tryCatch(
    httr::POST(
      groq_url,
      add_headers(
        Authorization = paste("Bearer", Sys.getenv("GROQ_API_KEY")),
        "Content-Type" = "application/json"
      ),
      body = jsonlite::toJSON(body, auto_unbox = TRUE),
      encode = "raw",
      httr::timeout(120)
    ),
    error = function(e) NULL
  )

  if (is.null(resp)) return("GROQ_REQUEST_ERROR")

  status <- httr::status_code(resp)
  if (status < 200 || status >= 300) return(paste0("GROQ_HTTP_", status))

  cont <- tryCatch(httr::content(resp, as = "parsed"), error = function(e) NULL)
  out  <- tryCatch(cont$choices[[1]]$message$content, error = function(e) NA_character_)

  out <- str_squish(as.character(out)[1])

  if (is.na(out) || !nzchar(out)) "EMPTY_MODEL_OUTPUT" else out
}

top_by_decade_groq <- top_by_decade %>%
  mutate(
    subject_groq = purrr::map_chr(text, ~ get_groq_subject(.x)),
    label_groq = glue("{speaker}\n{date}\n{subject_groq}")
  )

# -------------------------
# 7) Groq labeled plot (FIXED)
# -------------------------
message("Step 7: Creating Groq-labeled top-speeches plot...")

p_groq <- ggplot(speeches, aes(x = date, y = word_count)) +
  geom_point(
    aes(size = sqrt(word_count)),
    alpha = 0.08, color = "white",
    position = position_jitter(width = 0.3)
  ) +
  scale_size(range = c(1.2, 6), guide = "none") +
  scale_y_log10(labels = comma) +
  scale_x_date(date_breaks = "5 years", date_labels = "%Y") +

  geom_point(
    data = top_by_decade_groq,
    aes(size = sqrt(word_count)),
    color = "white"
  ) +

  geom_label_repel(
    data = top_by_decade_groq,
    aes(label = label_groq),
    fill = "white",
    color = "black",
    size = 2.5,
    box.padding = 1,
    point.padding = 0.5,
    force = 3,
    max.overlaps = Inf,
    min.segment.length = 0,
    segment.color = "white",        
    segment.curvature = -0.2,
    segment.angle = 15,
    segment.ncp = 8
  ) +

  labs(
    title = "Longest Speeches by Decade (Groq Subjects)",
    subtitle = "Top speeches per decade",
    x = "Date",
    y = "Word Count (log10)"
  ) +
  base_theme

ggsave(out_groq, p_groq, width = 28, height = 18, dpi = 600)
message("  - Saved plot: ", out_groq)

# -------------------------
# 7.5) Combined plot: TF-IDF dots + Groq labels
# -------------------------
message("Step 7.5: Creating TF-IDF dot plot WITH Groq labels...")

# Merge TF-IDF top speeches with Groq subjects
top_combined_labels <- top_by_decade_tfidf %>%
  left_join(top_by_decade_groq %>% select(speech_id, subject_groq),
            by = "speech_id") %>%
  mutate(label_groq_on_tfidf = glue("{speaker}\n{date}\n{subject_groq}"))

p_tfidf_with_groq <- ggplot(speeches, aes(x = date, y = word_count)) +
  geom_point(
    aes(size = sqrt(word_count)),
    alpha = 0.08, color = "white",
    position = position_jitter(width = 0.3)
  ) +
  scale_size(range = c(1.2, 6), guide = "none") +
  scale_y_log10(labels = comma) +
  scale_x_date(date_breaks = "5 years", date_labels = "%Y") +

  geom_point(
    data = top_combined_labels,
    aes(size = sqrt(word_count)),
    color = "white"
  ) +

  geom_label_repel(
    data = top_combined_labels,
    aes(label = label_groq_on_tfidf),
    fill = "white",
    color = "black",
    size = 2.5,
    box.padding = 1,
    point.padding = 0.5,
    force = 3,
    max.overlaps = Inf,
    min.segment.length = 0,
    segment.color = "white",
    segment.curvature = -0.2,
    segment.angle = 15,
    segment.ncp = 8
  ) +

  labs(
    title = "Longest Speeches by Decade",
    subtitle = "TF-IDF Dot Distribution with LLM Topic Labels",
    x = "Date",
    y = "Word Count (log10)"
  ) +
  base_theme

out_tfidf_groq <- file.path(output_dir, "dotplot_tfidf_with_groq_labels.png")
ggsave(out_tfidf_groq, p_tfidf_with_groq, width = 28, height = 18, dpi = 600)
message("  - Saved plot: ", out_tfidf_groq)

# -------------------------
# 8) Save CSV
# -------------------------
top_table <- top_by_decade %>%
  left_join(subjects_tfidf, by="speech_id") %>%
  left_join(top_by_decade_groq %>% select(speech_id, subject_groq), by="speech_id")

write.csv(top_table, out_top_csv, row.names = FALSE)

message("Done.")
