# ============================================================
# ko_spyRegression.R
# KO / SPY Historical Relationship Analysis
# Fully self-contained - fetches data via quantmod, no Python needed
#
# Daily data:   5 years from Yahoo → beta/corr by year and VIX regime
# Intraday:     60 days of 5-min from Yahoo → session breakdown
#
# Packages (run once):
#   install.packages(c("quantmod","tidyverse","zoo","scales","patchwork"))
# ============================================================

library(quantmod)
library(tidyverse)
library(zoo)
library(scales)
library(patchwork)

options(xts.warn_dplyr_breaks_lag = FALSE)

YEARS      <- 5
CHART_FILE <- "ko_spy_charts.pdf"
START_DAILY <- Sys.Date() - YEARS * 365

VIX_BREAKS <- c(0, 15, 25, 40, Inf)
VIX_LABELS <- c("Low (<15)", "Medium (15-25)", "High (25-40)", "Crisis (40+)")

# ── Helper ────────────────────────────────────────────────────
get_beta_corr <- function(x, y) {
  ok <- complete.cases(x, y)
  if (sum(ok) < 10 || sd(x[ok]) == 0) return(list(beta = NA_real_, corr = NA_real_))
  fit <- lm(y[ok] ~ x[ok])
  list(
    beta = round(unname(coef(fit)[2]), 3),
    corr = round(cor(x[ok], y[ok]), 3)
  )
}

assign_session <- function(t_str) {
  case_when(
    t_str >= "09:30" & t_str <= "11:29" ~ "Morning",
    t_str >= "11:30" & t_str <= "13:59" ~ "Midday",
    t_str >= "14:00" & t_str <= "15:55" ~ "Afternoon",
    TRUE ~ NA_character_
  )
}

# ════════════════════════════════════════════════════════════
# 1. DAILY DATA - 5 years (regime, year, rolling beta)
# ════════════════════════════════════════════════════════════
cat("Fetching 5-year daily data from Yahoo...\n")

getSymbols(c("KO", "SPY", "^VIX"),
           src = "yahoo",
           from = as.character(START_DAILY),
           auto.assign = TRUE)

# Chain two-object merges - xts join="inner" only works pairwise
merged_xts <- merge(Cl(KO), Cl(SPY), join = "inner")
merged_xts <- merge(merged_xts, Cl(VIX), join = "inner")
colnames(merged_xts) <- c("ko_close", "spy_close", "vix")

daily <- data.frame(
  date = as.Date(index(merged_xts)),
  coredata(merged_xts)
) |>
  filter(!is.na(ko_close), !is.na(spy_close)) |>
  arrange(date) |>
  mutate(
    ko_ret     = ko_close  / lag(ko_close)  - 1,
    spy_ret    = spy_close / lag(spy_close) - 1,
    year       = year(date),
    vix_regime = cut(vix, breaks = VIX_BREAKS, labels = VIX_LABELS,
                     right = FALSE, include.lowest = TRUE)
  ) |>
  filter(!is.na(ko_ret))

cat(sprintf("Daily: %s trading days | %s to %s\n\n",
            nrow(daily), min(daily$date), max(daily$date)))

# ── [1] By year ──────────────────────────────────────────────
tbl1 <- daily |>
  group_by(year) |>
  summarise(
    trading_days = n(),
    beta         = get_beta_corr(spy_ret, ko_ret)[["beta"]],
    correlation  = get_beta_corr(spy_ret, ko_ret)[["corr"]],
    ko_ann_ret   = paste0(round((prod(1 + ko_ret) ^ (252 / n()) - 1) * 100, 1), "%"),
    spy_ann_ret  = paste0(round((prod(1 + spy_ret) ^ (252 / n()) - 1) * 100, 1), "%"),
    .groups = "drop"
  )

cat("─── [1] Beta & Correlation by Year ──────────────────────\n")
print(tbl1)

# ── [2] By VIX regime ────────────────────────────────────────
tbl2 <- daily |>
  filter(!is.na(vix_regime)) |>
  group_by(vix_regime) |>
  summarise(
    trading_days = n(),
    beta         = get_beta_corr(spy_ret, ko_ret)[["beta"]],
    correlation  = get_beta_corr(spy_ret, ko_ret)[["corr"]],
    ko_avg_ret   = paste0(round(mean(ko_ret)  * 100, 4), "%"),
    spy_avg_ret  = paste0(round(mean(spy_ret) * 100, 4), "%"),
    .groups = "drop"
  )

cat("\n─── [2] Beta & Correlation by VIX Regime ────────────────\n")
print(tbl2)

# ── [3] Rolling 20-day beta ───────────────────────────────────
daily <- daily |>
  mutate(
    rolling_beta = rollapply(
      data      = cbind(spy_ret, ko_ret),
      width     = 20,
      FUN       = function(m) {
        if (any(is.na(m)) || sd(m[, 1]) == 0) return(NA)
        coef(lm(m[, 2] ~ m[, 1]))[2]
      },
      by.column = FALSE,
      fill      = NA,
      align     = "right"
    )
  )

tbl3 <- daily |>
  filter(!is.na(rolling_beta)) |>
  group_by(year) |>
  summarise(
    mean = round(mean(rolling_beta), 3),
    sd   = round(sd(rolling_beta),   3),
    min  = round(min(rolling_beta),  3),
    max  = round(max(rolling_beta),  3),
    .groups = "drop"
  )

cat("\n─── [3] Rolling 20-Day Beta Stats by Year ───────────────\n")
print(tbl3)

# ── [4] KO defensive outperformance days ──────────────────────
beta_full <- coef(lm(ko_ret ~ spy_ret, data = daily))[["spy_ret"]]

tbl4 <- daily |>
  filter(spy_ret < -0.005) |>
  mutate(
    expected_ko    = beta_full * spy_ret,
    outperformance = ko_ret - expected_ko
  ) |>
  arrange(desc(outperformance)) |>
  head(15) |>
  mutate(across(c(spy_ret, ko_ret, expected_ko, outperformance),
                ~ paste0(round(. * 100, 3), "%"))) |>
  select(date, spy_ret, ko_ret, expected_ko, outperformance)

cat("\n─── [4] KO Defensive Outperformance Days (SPY down >0.5%) ─\n")
print(as.data.frame(tbl4))

# ════════════════════════════════════════════════════════════
# 2. INTRADAY DATA - 60 days of 5-min (session breakdown)
# ════════════════════════════════════════════════════════════
cat("\nFetching 60-day 5-min intraday data from Yahoo...\n")

fetch_5min <- function(symbol) {
  x <- tryCatch(
    getSymbols(symbol, src = "yahoo", auto.assign = FALSE,
               periodicity = "5minutes",
               from = Sys.Date() - 60),
    error = function(e) { cat("  Warning:", conditionMessage(e), "\n"); NULL }
  )
  if (is.null(x)) return(NULL)
  data.frame(
    timestamp = index(x),
    close     = as.numeric(Cl(x)),
    symbol    = symbol
  )
}

spy5 <- fetch_5min("SPY")
ko5  <- fetch_5min("KO")

if (is.null(spy5) || is.null(ko5)) {
  cat("5-min data unavailable from Yahoo - session tables skipped.\n")
  cat("To unlock session analysis: run ko_spy_merged.py first to build the parquet cache,\n")
  cat("then switch to ko_spy_historical.R which reads that cache.\n\n")
} else {
  intra <- inner_join(
    spy5 |> rename(spy_close = close) |> select(-symbol),
    ko5  |> rename(ko_close  = close) |> select(-symbol),
    by = "timestamp"
  ) |>
    mutate(
      timestamp = as.POSIXct(timestamp, tz = "America/New_York"),
      date      = as.Date(timestamp),
      time_str  = format(timestamp, "%H:%M"),
      session   = assign_session(time_str)
    ) |>
    filter(!is.na(session)) |>
    arrange(timestamp) |>
    mutate(
      spy_ret = spy_close / lag(spy_close) - 1,
      ko_ret  = ko_close  / lag(ko_close)  - 1
    ) |>
    filter(!is.na(spy_ret))

  cat(sprintf("Intraday: %s 5-min bars | %s to %s\n\n",
              format(nrow(intra), big.mark = ","),
              min(intra$date), max(intra$date)))

  # [5] By session
  tbl5 <- intra |>
    group_by(session) |>
    summarise(
      bars        = n(),
      beta        = get_beta_corr(spy_ret, ko_ret)[["beta"]],
      correlation = get_beta_corr(spy_ret, ko_ret)[["corr"]],
      .groups = "drop"
    ) |>
    arrange(match(session, c("Morning", "Midday", "Afternoon")))

  cat("─── [5] Beta & Correlation by Session (60-day intraday) ─\n")
  print(tbl5)

  # [6] Session x VIX regime
  vix_map <- daily |> select(date, vix_regime)

  tbl6 <- intra |>
    left_join(vix_map, by = "date") |>
    filter(!is.na(vix_regime)) |>
    group_by(session, vix_regime) |>
    summarise(
      bars        = n(),
      beta        = get_beta_corr(spy_ret, ko_ret)[["beta"]],
      correlation = get_beta_corr(spy_ret, ko_ret)[["corr"]],
      .groups = "drop"
    ) |>
    arrange(match(session, c("Morning","Midday","Afternoon")), vix_regime)

  cat("\n─── [6] Session x VIX Regime (intraday) ─────────────────\n")
  print(tbl6, n = Inf)
}

# ════════════════════════════════════════════════════════════
# CHARTS
# ════════════════════════════════════════════════════════════
cat("\nBuilding charts →", CHART_FILE, "\n")

theme_clean <- theme_minimal(base_size = 11) +
  theme(
    plot.title       = element_text(face = "bold", size = 12),
    plot.subtitle    = element_text(color = "gray40", size = 10),
    panel.grid.minor = element_blank(),
    legend.position  = "bottom"
  )

p1 <- daily |>
  filter(!is.na(rolling_beta)) |>
  ggplot(aes(x = date, y = rolling_beta)) +
  geom_line(color = "#185FA5", linewidth = 0.6) +
  geom_smooth(method = "loess", span = 0.25, se = FALSE,
              color = "#D85A30", linetype = "dashed", linewidth = 1) +
  geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50") +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
  labs(title    = "KO/SPY Rolling 20-Day Beta",
       subtitle = "Dashed = LOESS trend | Dotted = beta 0.5 reference",
       x = NULL, y = "Beta") +
  theme_clean

p2 <- daily |>
  filter(!is.na(rolling_beta), !is.na(vix_regime)) |>
  ggplot(aes(x = vix_regime, y = rolling_beta, fill = vix_regime)) +
  geom_boxplot(alpha = 0.7, outlier.size = 0.5) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray40") +
  scale_fill_manual(values = c("#9FE1CB","#FAC775","#F09595","#AFA9EC")) +
  labs(title    = "KO Beta Distribution by VIX Regime",
       subtitle = "Beta compresses under stress - KO decouples from SPY",
       x = NULL, y = "Rolling 20-Day Beta") +
  theme_clean + theme(legend.position = "none")

p3 <- daily |>
  group_by(year) |>
  summarise(
    Correlation = cor(spy_ret, ko_ret, use = "complete.obs"),
    Beta        = get_beta_corr(spy_ret, ko_ret)[["beta"]],
    .groups = "drop"
  ) |>
  pivot_longer(c(Correlation, Beta), names_to = "metric", values_to = "value") |>
  ggplot(aes(x = factor(year), y = value, fill = metric)) +
  geom_col(position = "dodge", alpha = 0.85) +
  scale_fill_manual(values = c(Correlation = "#185FA5", Beta = "#D85A30")) +
  labs(title    = "KO/SPY Annual Beta & Correlation",
       subtitle = "Tracks relationship stability year over year",
       x = NULL, y = "Value", fill = NULL) +
  theme_clean

p4 <- daily |>
  filter(abs(spy_ret) > 0.002, !is.na(vix_regime)) |>
  ggplot(aes(x = spy_ret * 100, y = ko_ret * 100, color = vix_regime)) +
  geom_point(alpha = 0.3, size = 0.8) +
  geom_smooth(method = "lm", se = FALSE, color = "#185FA5", linewidth = 1) +
  scale_color_manual(values = c("#0F6E56","#BA7517","#D85A30","#7F77DD")) +
  labs(title    = "KO vs SPY Daily Returns by VIX Regime",
       subtitle = "Slope = full-history beta | Color = VIX regime at time of observation",
       x = "SPY daily return %", y = "KO daily return %", color = "VIX Regime") +
  theme_clean


# ════════════════════════════════════════════════════════════
# CO-MOVEMENT BACKTEST
# When KO and SPY move together (same direction, both >0.15%)
# during the decoupled regime (2024+, beta near 0), does SPY revert?
# ════════════════════════════════════════════════════════════

COMOVE_THRESHOLD <- 0.0015
REVERT_WINDOWS   <- c(1, 3, 6)

cat("\n--- [7] Co-Movement Backtest (2024+ regime, beta near 0) ---\n")

bt <- daily |>
  filter(year >= 2024) |>
  mutate(
    vix_high  = !is.na(vix) & vix >= 25,           # exclude high-VIX days from signal
    both_up   = spy_ret >  COMOVE_THRESHOLD & ko_ret >  COMOVE_THRESHOLD & !vix_high,
    both_down = spy_ret < -COMOVE_THRESHOLD & ko_ret < -COMOVE_THRESHOLD & !vix_high,
    comove    = both_up | both_down,
    direction = case_when(both_up ~ "up", both_down ~ "down", TRUE ~ NA_character_),
    row_n     = row_number()
  )

results <- list()
for (w in REVERT_WINDOWS) {
  signal_rows <- bt |> filter(comove) |> pull(row_n)
  dirs        <- bt |> filter(comove) |> pull(direction)

  fwd <- sapply(signal_rows, function(i) {
    end_i <- min(i + w, nrow(bt))
    if (end_i <= i) return(NA)
    sum(bt$spy_ret[(i + 1):end_i], na.rm = TRUE)
  })

  reverted <- ifelse(dirs == "up", fwd < 0, ifelse(dirs == "down", fwd > 0, NA))

  results[[paste0(w, "d")]] <- data.frame(
    window        = paste0(w, " day"),
    signals       = length(fwd),
    revert_rate   = round(mean(reverted, na.rm = TRUE) * 100, 1),
    avg_fwd_ret   = round(mean(fwd, na.rm = TRUE) * 100, 3),
    avg_when_rev  = round(mean(ifelse(reverted, fwd, NA), na.rm = TRUE) * 100, 3),
    avg_when_fail = round(mean(ifelse(!reverted, fwd, NA), na.rm = TRUE) * 100, 3)
  )
}

bt_tbl <- bind_rows(results)
colnames(bt_tbl) <- c("Window","Signals","Revert %","Avg fwd SPY %","Avg ret (rev)%","Avg ret (fail)%")
print(as.data.frame(bt_tbl))

cat("\n--- [8] Co-Movement by Direction ---\n")
dir_tbl <- bind_rows(lapply(REVERT_WINDOWS, function(w) {
  signal_rows <- bt |> filter(comove) |> pull(row_n)
  dirs        <- bt |> filter(comove) |> pull(direction)

  fwd <- sapply(signal_rows, function(i) {
    end_i <- min(i + w, nrow(bt))
    if (end_i <= i) return(NA)
    sum(bt$spy_ret[(i + 1):end_i], na.rm = TRUE)
  })

  reverted <- ifelse(dirs == "up", fwd < 0, ifelse(dirs == "down", fwd > 0, NA))

  data.frame(direction = dirs, fwd = fwd, reverted = reverted, window = paste0(w, " day")) |>
    group_by(window, direction) |>
    summarise(n = n(), revert_pct = round(mean(reverted, na.rm = TRUE) * 100, 1),
              avg_fwd = round(mean(fwd, na.rm = TRUE) * 100, 3), .groups = "drop")
}))
print(as.data.frame(dir_tbl))

cat("\n--- [9] Signal frequency by month ---\n")
freq_tbl <- bt |>
  filter(comove) |>
  mutate(month = format(date, "%Y-%m")) |>
  group_by(month, direction) |>
  summarise(signals = n(), .groups = "drop") |>
  arrange(month)
print(as.data.frame(freq_tbl))

# Chart 5: Signal scatter
signal_rows <- bt |> filter(comove) |> pull(row_n)
dirs        <- bt |> filter(comove) |> pull(direction)
dates_sig   <- bt |> filter(comove) |> pull(date)

fwd3 <- sapply(signal_rows, function(i) {
  end_i <- min(i + 3, nrow(bt))
  if (end_i <= i) return(NA)
  sum(bt$spy_ret[(i + 1):end_i], na.rm = TRUE)
})

plot_df <- data.frame(
  date      = dates_sig,
  direction = dirs,
  fwd3      = fwd3 * 100,
  reverted  = ifelse(dirs == "up", fwd3 < 0, fwd3 > 0)
) |> filter(!is.na(fwd3))

p5 <- ggplot(plot_df, aes(x = date, y = fwd3, color = reverted, shape = direction)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  geom_point(size = 2.5, alpha = 0.8) +
  scale_color_manual(values = c("TRUE" = "#0F6E56", "FALSE" = "#D85A30"),
                     labels = c("TRUE" = "Reverted", "FALSE" = "Failed")) +
  scale_shape_manual(values = c("up" = 24, "down" = 25)) +
  labs(title    = "KO/SPY Co-Movement Signals: 3-Day SPY Forward Return (2024+)",
       subtitle = "Green = SPY reversed as expected | Red = co-move continued",
       x = NULL, y = "SPY 3-day fwd return %", color = NULL, shape = "Direction") +
  theme_clean

# Chart 6: Cumulative strategy vs buy-hold
strat_rets <- rep(0, nrow(bt))
for (k in seq_along(signal_rows)) {
  i     <- signal_rows[k]
  end_i <- min(i + 3, nrow(bt))
  if (end_i <= i) next
  fwd_r <- sum(bt$spy_ret[(i + 1):end_i], na.rm = TRUE)
  strat_rets[i] <- if (dirs[k] == "up") -fwd_r else fwd_r
}

cum_df <- data.frame(
  date     = bt$date,
  strategy = cumprod(1 + strat_rets) - 1,
  buy_hold = cumprod(1 + bt$spy_ret) - 1
) |> pivot_longer(-date, names_to = "series", values_to = "cum_ret")

p6 <- ggplot(cum_df, aes(x = date, y = cum_ret * 100, color = series)) +
  geom_line(linewidth = 0.8) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  scale_color_manual(values = c(strategy = "#185FA5", buy_hold = "#888780"),
                     labels = c(strategy = "Fade co-move (3d)", buy_hold = "SPY buy & hold")) +
  labs(title    = "Co-Movement Fade Strategy vs SPY Buy & Hold (2024+)",
       subtitle = "Fades SPY for 3 days on every KO/SPY co-movement signal",
       x = NULL, y = "Cumulative return %", color = NULL) +
  theme_clean

pdf(CHART_FILE, width = 11, height = 8.5)
print(p1); print(p2); print(p3); print(p4); print(p5); print(p6)
dev.off()

cat("\nAll charts saved ->", CHART_FILE, "\n")

# ════════════════════════════════════════════════════════════
# PNG CHARTS — price_correlation.png and win_rate.png
# ════════════════════════════════════════════════════════════
dir.create("output/charts", recursive = TRUE, showWarnings = FALSE)

# ── price_correlation.png ─────────────────────────────────
# p_price: KO and SPY indexed to 100 at window start
price_indexed <- daily |>
  mutate(
    spy_idx = spy_close / first(spy_close) * 100,
    ko_idx  = ko_close  / first(ko_close)  * 100
  ) |>
  select(date, spy_idx, ko_idx) |>
  pivot_longer(c(spy_idx, ko_idx), names_to = "symbol", values_to = "value")

p_price <- ggplot(price_indexed, aes(x = date, y = value, color = symbol)) +
  geom_line(linewidth = 0.8) +
  scale_color_manual(values = c(spy_idx = "#185FA5", ko_idx = "#D85A30"),
                     labels  = c(spy_idx = "SPY",     ko_idx = "KO")) +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
  labs(title    = "KO vs SPY Indexed Price (base = 100)",
       subtitle = "5-year daily close, rebased to 100 at start of window",
       x = NULL, y = "Indexed price", color = NULL) +
  theme_clean

# p_corr_bar: rolling 20-day correlation bars + LOESS trend
rolling_corr <- daily |>
  mutate(
    rolling_corr = rollapply(
      data      = cbind(spy_ret, ko_ret),
      width     = 20,
      FUN       = function(m) cor(m[, 1], m[, 2], use = "complete.obs"),
      by.column = FALSE,
      fill      = NA,
      align     = "right"
    )
  ) |>
  filter(!is.na(rolling_corr))

p_corr_bar <- ggplot(rolling_corr, aes(x = date, y = rolling_corr)) +
  geom_col(aes(fill = rolling_corr > 0), alpha = 0.65, width = 2) +
  geom_smooth(method = "loess", span = 0.2, se = FALSE,
              color = "#185FA5", linewidth = 1) +
  geom_hline(yintercept = 0, color = "gray30") +
  scale_fill_manual(values = c("TRUE" = "#0F6E56", "FALSE" = "#D85A30"), guide = "none") +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
  labs(title    = "KO/SPY Rolling 20-Day Correlation",
       subtitle = "Green = positive | Red = negative | Blue line = LOESS trend",
       x = NULL, y = "Correlation") +
  theme_clean

p_combined <- p_price / p_corr_bar
ggsave("output/charts/price_correlation.png", p_combined,
       width = 10, height = 7, dpi = 150)
cat("Saved -> output/charts/price_correlation.png\n")

# ── win_rate.png ──────────────────────────────────────────
# Intraday signal win rates from Python backtest (last 365 days)
# Values from ko_spy_intraday.py: S1/S2/S4 × 0DTE and 3DTE windows
win_data <- data.frame(
  signal  = rep(c("S1: Both Down Fade",
                  "S4: SPY Up / KO Down",
                  "S2: SPY Down / KO Up"), each = 2),
  window  = rep(c("0DTE", "3DTE"), 3),
  win_pct = c(64.5, 67.3,   # S1
              63.9, 63.9,   # S4
              52.9, 67.8)   # S2
)
win_data$signal <- factor(win_data$signal,
                           levels = c("S1: Both Down Fade",
                                      "S4: SPY Up / KO Down",
                                      "S2: SPY Down / KO Up"))

p_win_rate <- ggplot(win_data, aes(x = signal, y = win_pct, fill = window)) +
  geom_col(position = "dodge", alpha = 0.85) +
  geom_text(aes(label = sprintf("%.1f%%", win_pct)),
            position = position_dodge(width = 0.9),
            vjust = -0.4, size = 3.5) +
  geom_hline(yintercept = 50, linetype = "dashed", color = "gray50") +
  scale_fill_manual(values = c("0DTE" = "#D85A30", "3DTE" = "#185FA5")) +
  scale_y_continuous(limits = c(0, 80), labels = function(x) paste0(x, "%")) +
  labs(title    = "KO/SPY Intraday Signal Win Rates by Hold Period",
       subtitle  = "Last 365 days | comparing same-day closes vs 3 DTE holds",
       x = NULL, y = "Win rate", fill = "Window") +
  theme_clean +
  theme(axis.text.x = element_text(size = 10))

ggsave("output/charts/win_rate.png", p_win_rate,
       width = 9, height = 6, dpi = 150)
cat("Saved -> output/charts/win_rate.png\n")

cat("Done.\n")
