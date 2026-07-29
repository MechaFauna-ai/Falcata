library(testthat)
library(falcata)  # nolint: unused_import.

test_check(
    package = "falcata"
    , stop_on_failure = TRUE
    , stop_on_warning = FALSE
    , reporter = testthat::SummaryReporter$new()
)
