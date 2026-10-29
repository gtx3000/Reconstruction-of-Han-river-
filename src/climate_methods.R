# Scientific methods used by scripts/run_climate_analysis.R.
# All inferential families, boundary rules and random seeds are explicit.
CLIMATE_SETTINGS <- list(wavelet_surrogates = 999L, seed = 20261028L,
  dj = 1 / 12, min_cycles = 3)

wavelet_cache_meta <- function(kind, inputs) {
  functions <- c("ar1_fit", "red_noise", "wavelet_fit", "trusted_wavelet")
  if (kind == "cwt") functions <- c(functions, "mc_p", "mc_cutoff", "compute_cwt")
  else functions <- c(functions, "coherence_fit", "simulate_wtc_null")
  if (kind == "wtc") functions <- c(functions, "mc_p", "mc_cutoff", "compute_wtc")
  # Deliberately exclude plotting functions: colour, size and DPI changes must
  # reuse the same statistical calculations.
  methods <- setNames(lapply(functions, function(name) {
    fun <- get(name, mode = "function")
    list(arguments = deparse(formals(fun)), implementation = deparse(body(fun)))
  }), functions)
  list(schema = 1L, kind = kind, inputs = inputs,
       settings = CLIMATE_SETTINGS[c("wavelet_surrogates", "seed", "dj", "min_cycles")],
       methods = methods, R = paste(R.version$major, R.version$minor, sep = "."),
       biwavelet = as.character(packageVersion("biwavelet")))
}

read_wavelet_cache <- function(path, metadata, required, force = FALSE) {
  if (force) {
    cat("  Force recomputation:", basename(path), "\n")
    return(NULL)
  }
  if (!file.exists(path)) {
    cat("  Cache missing:", basename(path), "\n")
    return(NULL)
  }
  value <- tryCatch(readRDS(path), error = function(e) NULL)
  if (!is.list(value)) {
    cat("  Cache unreadable; recomputing:", basename(path), "\n")
    return(NULL)
  }
  if (is.null(value$cache_metadata)) {
    cat("  Legacy cache has no input/parameter metadata; recomputing once:", basename(path), "\n")
    return(NULL)
  }
  if (!all(required %in% names(value))) {
    cat("  Cache incomplete; recomputing:", basename(path), "\n")
    return(NULL)
  }
  if (!identical(value$cache_metadata, metadata)) {
    cat("  Cache inputs, settings or calculation version changed:", basename(path), "\n")
    return(NULL)
  }
  cat("  Using cached result:", basename(path), "\n")
  value
}

check_annual <- function(years, values, label) {
  if (length(years) != length(values) || length(years) < 30 ||
      anyNA(years) || any(!is.finite(values)) || any(diff(years) != 1) ||
      sd(values) <= 0) stop(label, ": require finite, contiguous annual observations")
}

paired_data <- function(years, runoff, clim, label) {
  common <- sort(intersect(years, clim$years))
  x <- runoff[match(common, years)]
  y <- clim$vals[match(common, clim$years)]
  check_annual(common, x, paste(label, "runoff"))
  check_annual(common, y, label)
  list(years = common, runoff = x, climate = y)
}

ar1_fit <- function(x) {
  as.numeric(stats::arima(as.numeric(scale(x)), order = c(1, 0, 0),
                         include.mean = TRUE, method = "ML")$coef["ar1"])
}

red_noise <- function(n, phi) {
  as.numeric(scale(stats::arima.sim(model = list(ar = phi), n = n,
                                   n.start = max(200L, ceiling(20 / (1 - abs(phi)))))))
}

wavelet_fit <- function(x, years) {
  biwavelet::wt(cbind(years, as.numeric(scale(x))), mother = "morlet",
    param = 6, dj = CLIMATE_SETTINGS$dj, s0 = 2, pad = TRUE, do.sig = FALSE)
}

trusted_wavelet <- function(z) {
  outer(z$period, z$coi, `<`) &
    matrix(z$period <= length(z$t) / CLIMATE_SETTINGS$min_cycles,
           nrow = length(z$period), ncol = length(z$t))
}

coherence_fit <- function(x, y, years) {
  a <- wavelet_fit(x, years)
  b <- wavelet_fit(y, years)
  inv <- matrix(1 / a$scale, nrow(a$wave), ncol(a$wave))
  sm <- function(v) biwavelet::smooth.wavelet(v, 1, CLIMATE_SETTINGS$dj, a$scale)
  cross <- a$wave * Conj(b$wave)
  r2 <- abs(sm(inv * cross))^2 /
    (sm(inv * abs(a$wave)^2) * sm(inv * abs(b$wave)^2))
  r2[] <- pmin(1, pmax(0, r2))
  z <- list(rsq = r2, phase = Arg(cross), coi = pmin(a$coi, b$coi),
    period = a$period, scale = a$scale, t = years, xaxis = years,
    type = "wtc", sigma2 = 1)
  class(z) <- "biwavelet"
  z
}

mc_p <- function(observed, maxima) {
  vapply(observed, function(v) if (is.finite(v))
    (1 + sum(maxima >= v)) / (length(maxima) + 1) else NA_real_, numeric(1))
}

mc_cutoff <- function(maxima, alpha = .05) {
  # Match the discrete (1 + exceedances)/(B + 1) test, including Bonferroni.
  allowed_exceedances <- floor(alpha * (length(maxima) + 1) - 1 + 1e-12)
  if (allowed_exceedances < 0) return(Inf)
  sort(maxima)[length(maxima) - allowed_exceedances]
}

draw_wavelet <- function(z, label, local_cut = NULL, corrected_cut = NULL,
                         phase = FALSE, show_key = TRUE, period_limits = NULL,
                         period_breaks = NULL, x_breaks = NULL,
                         power_display = "power.norm") {
  trusted <- trusted_wavelet(z)
  if (phase) {
    z$phase[!trusted | z$rsq < local_cut] <- NA_real_
  }
  # Base graphics avoid biwavelet's colour-bar function resetting panel layout.
  colors <- colorRampPalette(c("#00007F", "blue", "#007FFF", "cyan",
                               "#7FFF7F", "yellow", "#FF7F00", "red", "#7F0000"))(64)
  if (phase) {
    values <- z$rsq
    limits <- c(0, 1)
  } else if (power_display == "power.corr.norm") {
    # Match the original biwavelet plotting default and its automatic colour
    # range. This changes the displayed power layer, not the tests or spectrum.
    values <- log2(pmax(z$power.corr / z$sigma2, .Machine$double.xmin))
    limits <- range(c(-1, 1) * max(values))
  } else {
    values <- log2(pmax(z$power / z$sigma2, 1e-6))
    limits <- log2(c(.25, 16))
  }
  values[] <- pmin(limits[2], pmax(limits[1], values))
  if (is.null(period_limits)) period_limits <- range(z$period)
  image(z$t, log2(z$period), t(values), col = colors, zlim = limits,
    ylim = rev(log2(period_limits)), xlab = "Year (CE)", ylab = "Period (yr)",
    main = label, yaxt = "n", xaxt = if (is.null(x_breaks)) "s" else "n",
    useRaster = TRUE)
  if (is.null(period_breaks)) {
    at <- axTicks(2); axis(2, at = at, labels = round(2^at))
  } else axis(2, at = log2(period_breaks), labels = period_breaks)
  if (!is.null(x_breaks)) axis(1, at = x_breaks, labels = x_breaks)
  bottom <- max(log2(period_limits))
  polygon(c(z$t, rev(z$t)), c(log2(z$coi), rep(bottom, length(z$t))),
           col = adjustcolor("white", .55), border = NA)
  lines(z$t, log2(z$coi), col = "grey50")
  # A colour key in the margin preserves mfrow/layout for subsequent panels.
  if (show_key) {
    width <- diff(range(z$t)); left <- max(z$t) + .025 * width
    right <- max(z$t) + .045 * width
    yy <- seq(bottom, min(log2(period_limits)), length.out = length(colors) + 1)
    rect(left, head(yy, -1), right, tail(yy, -1), col = colors, border = NA, xpd = NA)
    ticks <- seq(0, 1, length.out = 3)
    text(right + .015 * width, bottom + ticks * (min(log2(period_limits)) - bottom),
         labels = if (phase) ticks else signif(2^(limits[1] + ticks * diff(limits)), 2),
         cex = .7, xpd = NA, adj = 0)
  }
  field <- if (phase) z$rsq else z$ratio
  masked <- field
  masked[!trusted] <- NA_real_
  add_contour <- function(cut, col, lty, lwd) {
    if (!is.null(cut) && any(masked > cut, na.rm = TRUE) &&
        any(masked < cut, na.rm = TRUE))
      contour(z$t, log2(z$period), t(masked / cut), levels = 1, add = TRUE,
              drawlabels = FALSE, col = col, lty = lty, lwd = lwd)
  }
  add_contour(local_cut, "grey20", 2, 1.1)
  add_contour(corrected_cut, "black", 1, 2)
  if (phase) biwavelet::phase.plot(z$t, log2(z$period), z$phase,
    arrow.len = min(par()$pin[2] / 30, par()$pin[1] / 40),
    arrow.lwd = .02, arrow.col = "black")
  box()
}

plot_cwt_result <- function(r, site, out_dir) {
  z <- r$z
  period_limits <- c(2, 256)
  period_breaks <- c(4, 16, 64, 256)
  good <- r$table$n_inside_coi > 0 & z$period >= 2 & z$period <= 256
  g <- r$table$global_power_inside_coi
  threshold <- r$table$red_noise_background * r$global_fwer_cut
  # Match the original shallow, wide CWT layout (15 x 3 inches).
  png(file.path(out_dir, paste0(site, "_wavelet_power.png")),
      width = 15, height = 3, units = "in", res = 600)
  on.exit(dev.off(), add = TRUE)
  layout(matrix(c(1, 2), nrow = 1), widths = c(5.1, 1))
  par(mar = c(3.4, 3.4, .7, .8), mgp = c(2, .55, 0),
      cex.axis = .85, cex.lab = 1, font.lab = 2, tcl = -.25)
  first_tick <- ceiling((min(z$t) - 50) / 100) * 100 + 50
  middle_ticks <- if (first_tick <= max(z$t)) seq(first_tick, max(z$t), by = 100) else numeric()
  ticks <- sort(unique(c(min(z$t), middle_ticks)))
  ticks <- ticks[ticks >= min(z$t) & ticks <= max(z$t)]
  draw_wavelet(z, "", r$local_cut, r$fwer_cut,
    show_key = FALSE, period_limits = period_limits,
    period_breaks = period_breaks, x_breaks = ticks,
    power_display = "power.corr.norm")
  par(mar = c(3.4, 2.4, .7, .8))
  xvalues <- c(1, g[good], threshold[good])
  xmax <- ceiling(max(xvalues[is.finite(xvalues)]))
  plot(g[good], log2(z$period[good]), type = "l", lwd = 1.2,
    ylim = rev(log2(period_limits)), xlim = c(0, xmax),
    yaxt = "n", xaxt = "n", xlab = "Global Power", ylab = "")
  lines(threshold[good], log2(z$period[good]), col = "blue", lty = 2, lwd = 1.2)
  axis(1, at = seq_len(xmax))
  axis(2, at = log2(period_breaks), labels = period_breaks)
  invisible(NULL)
}

compute_cwt <- function(runoff, years) {
  cat("CWT: AR(1) Monte Carlo maximum statistics\n")
  phi <- ar1_fit(runoff)
  z <- wavelet_fit(runoff, years)
  trusted <- trusted_wavelet(z)
  bg <- (1 - phi^2) / (1 - 2 * phi * cos(2 * pi / z$period) + phi^2)
  ratio <- z$power / matrix(bg, nrow(z$power), ncol(z$power))
  good <- rowSums(trusted) > 0
  masked_mean <- function(a) rowSums(replace(a, !trusted, 0)) / rowSums(trusted)
  g <- masked_mean(z$power)
  gr <- masked_mean(ratio)
  B <- CLIMATE_SETTINGS$wavelet_surrogates
  maxima <- global_max <- numeric(B)
  set.seed(CLIMATE_SETTINGS$seed)
  for (b in seq_len(B)) {
    sim <- wavelet_fit(red_noise(length(years), phi), years)
    sr <- sim$power / matrix(bg, nrow(sim$power), ncol(sim$power))
    maxima[b] <- max(sr[trusted])
    global_max[b] <- max(masked_mean(sr)[good])
    if (b %% 100 == 0) cat("  CWT surrogate", b, "/", B, "\n")
  }
  cut <- mc_cutoff(maxima)
  gcut <- mc_cutoff(global_max)
  z$ratio <- ratio
  rows <- data.frame(period_years = z$period, n_inside_coi = rowSums(trusted),
    global_power_inside_coi = g, red_noise_background = bg,
    global_ratio = gr, global_p_fwer = mc_p(gr, global_max),
    n_local_sig = rowSums(trusted & ratio > qchisq(.95, 2) / 2),
    n_fwer_sig = rowSums(trusted & ratio > cut))
  list(phi = phi, z = z, table = rows, local_cut = qchisq(.95, 2) / 2,
       fwer_cut = cut, global_fwer_cut = gcut, surrogate_maxima = maxima)
}

run_cwt <- function(runoff, years, site, out_dir, force = FALSE) {
  check_annual(years, runoff, "CWT runoff")
  metadata <- wavelet_cache_meta("cwt",
    list(years = as.integer(years), runoff = as.numeric(runoff)))
  path <- file.path(out_dir, paste0(site, "_cwt.rds"))
  result <- read_wavelet_cache(path, metadata,
    c("phi", "z", "table", "local_cut", "fwer_cut", "global_fwer_cut", "surrogate_maxima"), force)
  if (is.null(result)) {
    result <- compute_cwt(runoff, years)
    result$cache_metadata <- metadata
    saveRDS(result, path)
  }
  write.csv(result$table, file.path(out_dir, paste0(site, "_cwt_summary.csv")), row.names = FALSE)
  plot_cwt_result(result, site, out_dir)
  invisible(result)
}

simulate_wtc_null <- function(pair, phi, z, map_index, mode) {
  B <- CLIMATE_SETTINGS$wavelet_surrogates
  trusted <- trusted_wavelet(z)
  maxima <- numeric(B)
  local_values <- vector("list", length(z$period))
  set.seed(CLIMATE_SETTINGS$seed + map_index)
  for (b in seq_len(B)) {
    sim <- coherence_fit(red_noise(length(pair$years), phi[1]),
                         red_noise(length(pair$years), phi[2]), pair$years)
    maxima[b] <- max(sim$rsq[trusted])
    for (s in which(rowSums(trusted) > 0)) {
      idx <- which(trusted[s, ])
      idx <- idx[unique(round(seq(1, length(idx), length.out = min(12, length(idx)))))]
      local_values[[s]] <- c(local_values[[s]], sim$rsq[s, idx])
    }
    if (b %% 100 == 0) cat("  WTC", mode, "surrogate", b, "/", B, "\n")
  }
  local <- vapply(local_values, function(v) if (length(v))
    unname(quantile(v, .95, type = 1)) else NA_real_, numeric(1))
  list(maxima = maxima, local = local, phi = phi, settings = CLIMATE_SETTINGS)
}

compute_wtc <- function(years, runoff, climate, site, out_dir, force = FALSE) {
  modes <- names(climate)
  B <- CLIMATE_SETTINGS$wavelet_surrogates
  fits <- list(); max_by_mode <- matrix(NA_real_, B, length(modes))
  ar_table <- list()
  for (i in seq_along(modes)) {
    m <- modes[i]
    pair <- paired_data(years, runoff, climate[[m]], m)
    phi <- c(ar1_fit(pair$runoff), ar1_fit(pair$climate))
    z <- coherence_fit(pair$runoff, pair$climate, pair$years)
    metadata <- wavelet_cache_meta("wtc_null", list(pair = pair, map_index = i, mode = m))
    path <- file.path(out_dir, paste0(site, "_wtc_null_", m, ".rds"))
    null <- read_wavelet_cache(path, metadata, c("maxima", "local", "phi"), force)
    if (is.null(null)) {
      null <- simulate_wtc_null(pair, phi, z, i, m)
      null$cache_metadata <- metadata
      saveRDS(null, path)
    }
    max_by_mode[, i] <- null$maxima
    fits[[m]] <- list(z = z, pair = pair, phi = phi, local = null$local)
    ar_table[[m]] <- data.frame(mode = m, start = min(pair$years), end = max(pair$years),
      n = length(pair$years), runoff_ar1 = phi[1], climate_ar1 = phi[2])
  }
  cuts <- apply(max_by_mode, 2, mc_cutoff, alpha = .05 / length(modes))
  summaries <- list(); regions <- list()
  for (m in modes) {
    i <- match(m, modes)
    cut <- unname(cuts[i])
    z <- fits[[m]]$z
    local_mat <- matrix(fits[[m]]$local, nrow(z$rsq), ncol(z$rsq))
    inside <- trusted_wavelet(z)
    fwer <- inside & z$rsq > cut
    nominal <- inside & z$rsq > local_mat
    nominal[is.na(nominal)] <- FALSE
    bands <- c(2, 8, 16, 32, 64, 128, Inf)
    summaries[[m]] <- do.call(rbind, lapply(seq_len(6), function(k) {
      mask <- inside & matrix(z$period >= bands[k] & z$period < bands[k + 1],
                                nrow(z$rsq), ncol(z$rsq))
      data.frame(mode = m, period_band_years = paste0(bands[k], "-", bands[k + 1]),
        n_inside_coi = sum(mask), n_local_sig = sum(nominal & mask),
        n_fwer_sig = sum(fwer & mask), max_coherence = if (any(mask)) max(z$rsq[mask]) else NA,
        p_fwer = if (any(mask)) min(1, length(modes) *
          mc_p(max(z$rsq[mask]), max_by_mode[, i])) else NA)
    }))
    if (any(fwer)) {
      ii <- which(fwer, arr.ind = TRUE)
      regions[[m]] <- data.frame(mode = m, year = z$t[ii[, 2]],
        period_years = z$period[ii[, 1]], coherence = z$rsq[ii], phase_radians = z$phase[ii])
    }
    fits[[m]]$fwer_cut <- cut
  }
  cells <- if (length(regions)) do.call(rbind, regions) else
    data.frame(mode = character(), year = integer(), period_years = numeric(),
               coherence = numeric(), phase_radians = numeric())
  list(fits = fits, summaries = do.call(rbind, summaries),
       overlap_ar1 = do.call(rbind, ar_table), fwer_cells = cells,
       map_maxima = max_by_mode, fwer_cuts = cuts,
       between_mode_adjustment = "Bonferroni")
}

plot_wtc_result <- function(r, site, out_dir) {
  png(file.path(out_dir, paste0(site, "_wtc_panel.png")),
      width = 16, height = 32 / 3, units = "in", res = 600)
  on.exit(dev.off(), add = TRUE)
  par(mfrow = c(2, 2), mar = c(4, 4, 3, 4))
  for (m in names(r$fits)) {
    f <- r$fits[[m]]
    local <- matrix(f$local, nrow(f$z$rsq), ncol(f$z$rsq))
    draw_wavelet(f$z, paste("Runoff and", m), local, f$fwer_cut, phase = TRUE)
  }
  invisible(NULL)
}

run_wtc <- function(years, runoff, climate, site, out_dir, force = FALSE) {
  inputs <- lapply(names(climate), function(m) {
    pair <- paired_data(years, runoff, climate[[m]], m)
    lapply(pair, as.numeric)
  })
  names(inputs) <- names(climate)
  metadata <- wavelet_cache_meta("wtc", inputs)
  path <- file.path(out_dir, paste0(site, "_wtc.rds"))
  result <- read_wavelet_cache(path, metadata,
    c("fits", "summaries", "overlap_ar1", "fwer_cells", "map_maxima", "fwer_cuts"), force)
  if (is.null(result)) {
    result <- compute_wtc(years, runoff, climate, site, out_dir, force)
    result$cache_metadata <- metadata
    saveRDS(result, path)
  }
  write.csv(result$summaries, file.path(out_dir, paste0(site, "_wtc_coi_summary.csv")), row.names = FALSE)
  write.csv(result$overlap_ar1, file.path(out_dir, paste0(site, "_overlap_ar1.csv")), row.names = FALSE)
  write.csv(result$fwer_cells, file.path(out_dir, paste0(site, "_wtc_fwer_cells.csv")), row.names = FALSE)
  plot_wtc_result(result, site, out_dir)
  invisible(result)
}

render_saved_wavelets <- function(site, out_dir, cwt = TRUE, wtc = TRUE) {
  file <- file.path(out_dir, paste0(site, "_cwt.rds"))
  if (cwt && file.exists(file)) plot_cwt_result(readRDS(file), site, out_dir)
  file <- file.path(out_dir, paste0(site, "_wtc.rds"))
  if (wtc && file.exists(file)) plot_wtc_result(readRDS(file), site, out_dir)
  invisible(NULL)
}
