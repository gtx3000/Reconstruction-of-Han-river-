# CWT/WTC and extreme drought/flood frequency for the Hanjiang reconstruction.
# Methods: ../src/climate_methods.R; inferential settings are explicit there.
# Usage: Rscript scripts/run_climate_analysis.R [--skip-cwt] [--skip-wtc] [--force-recompute]
# Optional --results-dir and --climate-file override config/default.yaml.
# Figures, numerical summaries and a runtime log are saved under climate/.
suppressPackageStartupMessages({
  library(biwavelet)
  library(ggplot2)
  library(readxl)
  library(yaml)
})

# Derive the project root from the script's own location (works from any cwd)
get_project_root <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    script_path <- normalizePath(sub("^--file=", "", file_arg[1]),
                                 winslash = "/")
    return(dirname(dirname(script_path)))   # scripts/.. = project root
  }
  getwd()
}

PROJECT_ROOT <- get_project_root()
is_abs_path <- function(p) grepl("^[A-Za-z]:", p) || startsWith(p, "/") ||
  startsWith(p, intToUtf8(92))

# ==================== [Edit here] single-station info ====================
SITE_NAME <- "Hanjiang"   # station folder name
# ==========================================================================

# Read config/default.yaml (edit paths there; no code changes required)
load_config <- function() {
  cfg_file <- file.path(PROJECT_ROOT, "config", "default.yaml")
  if (!file.exists(cfg_file)) stop("Config file not found: ", cfg_file)
  yaml::read_yaml(cfg_file)
}

# ---------------- Command-line arguments ----------------
parse_args <- function() {
  cfg <- load_config()
  paths <- cfg$paths

  # Defaults come from default.yaml; fall back to "results" if missing
  out_dir <- if (is.null(paths$output_dir) || nchar(paths$output_dir) == 0)
    "results" else paths$output_dir
  clim_file <- paths$climate_file
  if (is.null(clim_file) || nchar(clim_file) == 0)
    stop("Please set paths.climate_file in config/default.yaml")

  a <- commandArgs(trailingOnly = TRUE)
  opts <- list(results_dir = out_dir,
               climate_file = clim_file,
               skip_cwt = FALSE, skip_wtc = FALSE, force_recompute = FALSE)
  i <- 1
  while (i <= length(a)) {
    key <- a[i]
    if (key == "--results-dir")  { opts$results_dir  <- a[i + 1]; i <- i + 2 }
    else if (key == "--climate-file") { opts$climate_file <- a[i + 1]; i <- i + 2 }
    else if (key == "--skip-cwt")  { opts$skip_cwt  <- TRUE; i <- i + 1 }
    else if (key == "--skip-wtc")  { opts$skip_wtc  <- TRUE; i <- i + 1 }
    else if (key == "--force-recompute") { opts$force_recompute <- TRUE; i <- i + 1 }
    else { stop("Unknown option: ", key) }
  }
  opts
}

# ---------------- Utility functions ----------------

read_recon_file <- function(path) {
  if (grepl("\\.xlsx$", path, ignore.case = TRUE)) {
    as.data.frame(read_excel(path))
  } else {
    tryCatch(read.csv(path, fileEncoding = "UTF-8-BOM", check.names = FALSE),
             error = function(e) read.csv(path, check.names = FALSE))
  }
}

load_climate_indices <- function(climate_file) {
  sheets <- c(ENSO = "enso", NAO = "nao", PDO = "pdo", IOD = "iod")
  out <- list()
  for (m in names(sheets)) {
    df <- as.data.frame(read_excel(climate_file, sheet = sheets[[m]]))
    years <- as.integer(df[[1]])
    vals  <- as.numeric(df[[2]])
    ok <- !is.na(years) & !is.na(vals)
    out[[m]] <- list(years = years[ok], vals = vals[ok])
  }
  out
}

extreme_freq <- function(years, series, window) {
  high <- mean(series, na.rm = TRUE) + sd(series, na.rm = TRUE)
  low  <- mean(series, na.rm = TRUE) - sd(series, na.rm = TRUE)
  y0 <- as.integer(years[1]); y1 <- as.integer(years[length(years)])

  # Window alignment: the first window may be incomplete (e.g. 960-999 for a
  # 50-yr window is only 40 years); remaining windows start aligned to window;
  # a trailing partial window is dropped.
  if (y0 %% window == 0L) {
    starts <- seq(y0, y1, by = window)
    ends   <- pmin(starts + window - 1L, y1)
  } else {
    nxt <- (y0 %/% window + 1L) * window          # e.g. 960 -> 1000
    starts <- c(y0, seq(nxt, y1, by = window))
    ends   <- c(nxt - 1L, pmin(starts[-1] + window - 1L, y1))
  }
  n_years <- ends - starts + 1L
  keep <- n_years >= window | seq_along(starts) == 1L
  starts <- starts[keep]; ends <- ends[keep]; n_years <- n_years[keep]

  rows <- lapply(seq_along(starts), function(i) {
    seg <- series[years >= starts[i] & years <= ends[i]]
    seg <- seg[!is.na(seg)]
    if (length(seg) == 0) return(NULL)
    data.frame(year_start = starts[i], year_end = ends[i],
               n_years = n_years[i],
               wet = sum(seg > high), dry = sum(seg < low))
  })
  do.call(rbind, Filter(Negate(is.null), rows))
}

plot_freq <- function(site, freq_df, window, title_txt, out_png) {
  long_df <- data.frame(
    year  = rep(freq_df$year_start, 2),
    num   = c(freq_df$wet, -freq_df$dry),
    class = rep(c("wet", "dry"), each = nrow(freq_df)),
    width = rep(freq_df$n_years * if (window <= 10) 0.9 else 0.4, 2)
  )
  ymax <- max(c(freq_df$wet, freq_df$dry), 1) + 1
  step <- max(1, round(2 * ymax / 8))

  # x-axis ticks: 10-yr figure uses bar start years (first/last always, plus
  # every 100 years in between); 50-yr figure labels every bar (960, 1000, ...)
  starts <- freq_df$year_start
  first_s <- starts[1]; last_s <- starts[length(starts)]
  if (window <= 10) {
    x_breaks <- starts[starts == first_s |(starts - first_s) %% 100 == 0]
    x_labels <- paste0(x_breaks, "s")
  } else {
    x_breaks <- starts
    x_labels <- as.character(x_breaks)
  }

  p <- ggplot(long_df, aes(x = year, y = num, fill = class)) +
    geom_bar(stat = "identity", color = "white",
             aes(width = width), alpha = 0.9) +
    geom_hline(yintercept = 0, linewidth = 0.8, color = "black") +
    scale_fill_manual("", values = c(dry = "#C34128", wet = "#206095"),
                      labels = c("Dry", "Wet")) +
    scale_y_continuous(breaks = seq(-floor(ymax), floor(ymax), by = step),
                       labels = abs,
                       limits = c(-ceiling(ymax), ceiling(ymax)),
                       expand = c(0, 0)) +
    scale_x_continuous(breaks = x_breaks, labels = x_labels,
                       expand = c(0.01, 0)) +
    labs(x = "Year(CE)", y = "Count") +
    theme_minimal(base_family = "Arial") +
    theme(
      text = element_text(family = "Arial", size = 14),
      plot.title = element_text(hjust = 0.5, size = 24, face = "bold",
                                margin = margin(b = 15)),
      axis.title = element_text(size = 18, face = "bold"),
      axis.text = element_text(size = 14, color = "black"),
      axis.line = element_line(color = "black", linewidth = 0.4),
      axis.ticks = element_line(color = "black", linewidth = 0.5),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      legend.position = "top",
      legend.title = element_blank(),
      legend.text = element_text(size = 12),
      legend.margin = margin(b = 5),
      panel.border = element_rect(color = "black", fill = NA, linewidth = 0.4),
      plot.margin = margin(12, 12, 12, 12)
    )

  ggsave(out_png, p, width = 14, height = 5.5, dpi = 600, bg = "white")
  cat(sprintf("    ok extreme freq: %s\n", basename(out_png)))
}

main <- function() {
  opts <- parse_args()
  if (!is_abs_path(opts$results_dir)) opts$results_dir <- file.path(PROJECT_ROOT, opts$results_dir)
  if (!is_abs_path(opts$climate_file)) opts$climate_file <- file.path(PROJECT_ROOT, opts$climate_file)
  source(file.path(PROJECT_ROOT, "src", "climate_methods.R"))
  site_dir <- file.path(opts$results_dir, SITE_NAME)
  path <- file.path(site_dir, "table", "reconstructed_runoff_final.xlsx")
  if (!file.exists(path)) path <- sub("\\.xlsx$", ".csv", path)
  if (!file.exists(path)) stop("Missing reconstruction: ", path)
  out_dir <- file.path(site_dir, "climate")
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  log_path <- file.path(out_dir, "climate_analysis_run.log")
  con <- file(log_path, "at", encoding = "UTF-8")
  sink(con, split = TRUE)
  on.exit({sink(); close(con)}, add = TRUE)
  options(warn = 1)
  cat("\nRun:", format(Sys.time(), tz = "Asia/Shanghai", usetz = TRUE),
      "|", paste(commandArgs(trailingOnly = TRUE), collapse = " "), "\n")
  d <- read_recon_file(path)
  years <- as.integer(d$year); runoff <- as.numeric(d$reconstructed_runoff)
  check_annual(years, runoff, "Reconstructed runoff")
  climate <- load_climate_indices(opts$climate_file)
  for (m in names(climate)) check_annual(climate[[m]]$years, climate[[m]]$vals, m)
  cat("CWT/WTC with AR(1) backgrounds; 10- and 50-year extreme drought/flood counts\n")
  if (!opts$skip_cwt) run_cwt(runoff, years, SITE_NAME, out_dir, force = opts$force_recompute)
  if (!opts$skip_wtc) run_wtc(years, runoff, climate, SITE_NAME, out_dir, force = opts$force_recompute)
  # Original extreme drought/flood definition and window alignment.
  for (window in c(10L, 50L)) {
    freq_df <- extreme_freq(years, runoff, window)
    plot_freq(SITE_NAME, freq_df, window,
      sprintf("Runoff %d-yr (%d-%d)", window,
              freq_df$year_start[1], freq_df$year_end[nrow(freq_df)]),
      file.path(out_dir, sprintf("%s_extreme_freq_%dyr.png", SITE_NAME, window)))
  }
  render_saved_wavelets(SITE_NAME, out_dir, cwt = opts$skip_cwt, wtc = opts$skip_wtc)
  cat("Completed all requested analyses.\n")
}

if (sys.nframe() == 0L) main()
