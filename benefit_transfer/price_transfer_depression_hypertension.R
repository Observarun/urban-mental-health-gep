library(readxl)  # reads the r250 country list (xlsx)
library(sandwich)  # vcovCL(): standard errors that allow for correlation within a country
library(lmtest)  # coeftest(), coefci(): coefficient table and confidence intervals

# ----------------------------------------------------------------------------------
# 0. Where the files are.
# ----------------------------------------------------------------------------------
DATA_DIR <- "~/"
OUT_DIR  <- "~/"
DEP_FILE <- paste0(DATA_DIR, "mental_health_cost_2019usd.csv")  # depression, 2019 USD
HT_FILE  <- paste0(DATA_DIR, "ht_costs_uv.csv")  # hypertension, 2019 USD
IDS_FILE <- paste0(DATA_DIR, "EXAMPLE_SERVICE_prices_raw_data.xlsx")  # iso3_r250_id / label / name
GDP_FILE <- paste0(DATA_DIR, "API_NY_GDP_PCAP_CD_DS2_en_csv_v2_1529.csv")  # GDP per capita, current US$

# World Bank csv files have 4 header lines before the real column names.
# gdp_2019 is a vector of 2019 GDP per capita named by ISO3 code: gdp_2019["KEN"] looks up Kenya,
# and a code that is not in the file gives NA (not an error).
gdp_df   <- read.csv(GDP_FILE, skip = 4, check.names = FALSE)
gdp_2019 <- setNames(as.numeric(gdp_df[["2019"]]), gdp_df[["Country Code"]])

# ----------------------------------------------------------------------------------
# 1. Data preparation.
# ----------------------------------------------------------------------------------
# 1a. The r250 country list (one row per country id)
ids <- as.data.frame(read_excel(IDS_FILE))[, c("iso3_r250_id", "iso3_r250_label", "iso3_r250_name")]
ids <- ids[!duplicated(ids$iso3_r250_id), ]

# look-up from country name to ISO3 code, used to attach a code to every cost row
# a name that appears twice in the list (e.g. "China") cannot identify one country & is left out
nm <- trimws(ids$iso3_r250_name)
ambiguous <- duplicated(nm) | duplicated(nm, fromLast = TRUE)
iso3_by_name <- setNames(ids$iso3_r250_label[!ambiguous], nm[!ambiguous])

to_iso3 <- function(country) {
  iso3 <- unname(iso3_by_name[trimws(country)])
  if (anyNA(iso3)) stop("country names not found in iso3_r250_name: ", paste(country[is.na(iso3)], collapse = ", "))
  iso3
}

# 1b. Read the two cost files.
#   depression:   total cost per patient (price_usd_2019)
#   hypertension: cost per person (total cost where available, else direct cost or a component of it)
dep_in <- read.csv(DEP_FILE, check.names = FALSE, stringsAsFactors = FALSE)
ht_in  <- read.csv(HT_FILE,  check.names = FALSE, stringsAsFactors = FALSE, fileEncoding = "UTF-8-BOM")

# some hypertension values have a thousands separator ("1,864.69"): remove the commas, then convert to numbers
ht_in$cost_2019USD <- as.numeric(gsub(",", "", ht_in$cost_2019USD))
stopifnot(all(dep_in$price_usd_2019 > 0), all(ht_in$cost_2019USD > 0))   # every row needs a positive 2019 USD cost

# 1c. Stack the two services into ONE dataset and build the regression variables.
stack <- rbind(
  data.frame(iso3 = to_iso3(ht_in$country),  cost = ht_in$cost_2019USD,    service = "hypertension"),
  data.frame(iso3 = to_iso3(dep_in$country), cost = dep_in$price_usd_2019, service = "depression")
)
stack$ln_cost  <- log(stack$cost)
stack$ln_gdppc <- unname(log(gdp_2019[stack$iso3]))        # income = GDP per capita, current US$, 2019
stack$D_dep    <- as.integer(stack$service == "depression")  # service dummy: 1 = depression, 0 = hypertension
stopifnot(!anyNA(stack$ln_gdppc))                           # every country in the cost files needs a GDP per capita

cat("Observations and GDP per capita range (2019 US$) by service:\n")
print(do.call(rbind, lapply(split(stack, stack$service), function(d)
  data.frame(n = nrow(d), min_gdppc = round(exp(min(d$ln_gdppc))), max_gdppc = round(exp(max(d$ln_gdppc)))))))
cat("Countries (clusters):", length(unique(stack$iso3)), "\n")

# ----------------------------------------------------------------------------------
# 2. The three models, all with country-clustered standard errors.
# ----------------------------------------------------------------------------------
# A country can appear in both services, so its two observations are not independent.
# "Clustering by country" tells the standard errors about this.
cluster_ols <- function(formula, data) {
  fit <- lm(formula, data)
  vc  <- vcovCL(fit, cluster = data$iso3)  # applies small-sample corrections for few clusters
  df  <- length(unique(data$iso3)) - 1
  list(fit = fit, tab = cbind(coeftest(fit, vcov. = vc, df = df), coefci(fit, vcov. = vc, df = df)))
}  # p-values and CIs use a t distribution with (# countries - 1) DsOF

SHARED   <- ln_cost ~ D_dep + ln_gdppc    # common income slope, separate intercept per service
INTERACT <- ln_cost ~ D_dep * ln_gdppc    # also a separate slope per service (adds D_dep:ln_gdppc)

# separate: one regression per service, each with its own slope
cat("\n=== Separate regressions (one per service), country-clustered SEs ===\n")
services <- c("hypertension", "depression")
sep <- lapply(setNames(services, services), function(s) cluster_ols(ln_cost ~ ln_gdppc, stack[stack$service == s, ]))
print(do.call(rbind, lapply(sep, function(m) {
  b <- m$tab["ln_gdppc", ]
  data.frame(n = nobs(m$fit), slope = b[["Estimate"]], SE = b[["Std. Error"]],
             CI_low = b[["2.5 %"]], CI_high = b[["97.5 %"]], R2 = summary(m$fit)$r.squared)
})), digits = 3)

# shared: one common slope, depression gets own intercept through dummy
A <- cluster_ols(SHARED, stack)
cat("\n=== Shared-slope model (common slope, service dummy) ===\n")
print(round(A$tab, 4))
cat(sprintf("Depression / hypertension cost ratio at the same income = exp(d) = %.2f\n", exp(coef(A$fit)[["D_dep"]])))

# interaction: depression also gets its own slope
B <- cluster_ols(INTERACT, stack)
cat("\n=== Interaction model (separate slopes) ===\n")
print(round(B$tab, 4))

# ----------------------------------------------------------------------------------
# 3. Do the two services need different income slopes?
#  g = (depression slope) - (hypertension slope), the D_dep:ln_gdppc row of the interaction model
# ----------------------------------------------------------------------------------
cat("\n=== Slope difference g (depression - hypertension), cluster-robust ===\n")
print(round(B$tab["D_dep:ln_gdppc", , drop = FALSE], 3))

# ----------------------------------------------------------------------------------
# 4. Leave-one-country-out cross-validation: separate slopes vs shared slope.
#  for each country drop all its observations (both services), refit on rest, predict dropped observations' ln(cost)
#  whole countries are held out because a country's hypertension and depression costs aren't independent of each other
# ----------------------------------------------------------------------------------
cv <- do.call(rbind, lapply(unique(stack$iso3), function(country) {
  train <- stack[stack$iso3 != country, ]
  test  <- stack[stack$iso3 == country, ]
  data.frame(service    = test$service,
             err_sep    = test$ln_cost - predict(lm(INTERACT, train), test),
             err_shared = test$ln_cost - predict(lm(SHARED, train), test))
}))

rmse <- function(err) sqrt(mean(err^2))
cat("\n=== Leave-one-country-out cross-validation: RMSE in log points (lower = better) ===\n")
cv_table <- sapply(list(separate = cv$err_sep, shared = cv$err_shared), function(err)
  c(all = rmse(err), tapply(err, cv$service, rmse)))
print(round(t(cv_table), 3))
#  interaction model gives the same predictions as running the two services separately

# ----------------------------------------------------------------------------------
# 5. Predict cost for every country in the r250 list, for both services.
#  countries with no 2019 GDP per capita are retained with missing predictions
#  predicted cost = exp(predicted ln cost) = the conditional median
#  predictions come from the shared-slope model only (common income slope, service dummy)
# ----------------------------------------------------------------------------------
countries <- ids
countries$gdppc_usd_2019 <- unname(gdp_2019[countries$iso3_r250_label])
countries$ln_gdppc       <- log(countries$gdppc_usd_2019)
cat(sprintf("\n=== Predicting for %d of %d r250 countries (the rest have no 2019 GDP per capita in US$) ===\n",
            sum(!is.na(countries$ln_gdppc)), nrow(countries)))

out <- countries[, c("iso3_r250_id", "iso3_r250_label", "iso3_r250_name", "gdppc_usd_2019")]
out$cost_ht_shared  <- exp(predict(A$fit, transform(countries, D_dep = 0)))
out$cost_dep_shared <- exp(predict(A$fit, transform(countries, D_dep = 1)))

#  outside a service's observed income range a prediction is an extrapolation
#  the further outside less the trust
for (s in services) {
  rng <- range(stack$ln_gdppc[stack$service == s])
  n_in <- sum(countries$ln_gdppc >= rng[1] & countries$ln_gdppc <= rng[2], na.rm = TRUE)
  cat(sprintf("Countries inside the %s training range: %d of %d\n", s, n_in, sum(!is.na(countries$ln_gdppc))))
}

out[-(1:3)] <- round(out[-(1:3)], 2)
out <- out[order(out$iso3_r250_label, method = "radix"), ]
out_path <- paste0(OUT_DIR, "health_cost_predictions_2019usd.csv")
write.csv(out, out_path, row.names = FALSE, na = "", fileEncoding = "UTF-8")  # keeps accented names intact
cat(sprintf("\nSaved %d rows to %s\n", nrow(out), out_path))

# ----------------------------------------------------------------------------------
# 6. Leave-one-observation-out stability of the shared slope.
#  drop each observation in turn, refit the shared model, and see how far the common income slope moves
#  narrow range means no single observation drives the slope
# ----------------------------------------------------------------------------------
loo_slopes <- sapply(seq_len(nrow(stack)), function(i) coef(lm(SHARED, stack[-i, ]))[["ln_gdppc"]])
lo_i <- which.min(loo_slopes); hi_i <- which.max(loo_slopes)
cat(sprintf("\n=== Leave-one-observation-out: shared slope (full sample = %.3f) ===\n", coef(A$fit)[["ln_gdppc"]]))
cat(sprintf("  range %.3f to %.3f\n", min(loo_slopes), max(loo_slopes)))
cat(sprintf("  lowest  when dropping %s (%s)\n", stack$iso3[lo_i], stack$service[lo_i]))
cat(sprintf("  highest when dropping %s (%s)\n", stack$iso3[hi_i], stack$service[hi_i]))
