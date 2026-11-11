.bounds_model <- function() {
  lgb.train(
    params = list(
      objective = "binary"
      , metric = c("binary_logloss", "auc", "binary_error")
      , min_data_in_leaf = 1L
      , num_threads = 1L
      , verbosity = -1L
    )
    , data = lgb.Dataset(matrix(as.double(seq_len(80L)), ncol = 2L), label = rep(c(0.0, 1.0), 20L))
    , nrounds = 2L
  )
}

test_that("GetEval refuses an out_result buffer smaller than the evaluation count", {
  model <- .bounds_model()
  handle <- model$.__enclos_env__$private$handle
  expect_error(
    .Call(FLC_BoosterGetEval_R, handle, 0L, numeric(1L))
    , "out_result holds 1 values, but the booster has 3 evaluation results"
  )
})

test_that("sparse count arguments beyond the int32 range are refused, not turned into NA", {
  model <- .bounds_model()
  handle <- model$.__enclos_env__$private$handle
  indptr <- c(0L, 1L, 1L)
  expect_error(
    .Call(
      FLC_BoosterPredictForCSC_R, handle, indptr, 0L, 0.5, 3e9, 1L, 1L
      , FALSE, FALSE, FALSE, 0L, -1L, "", numeric(1L)
    )
    , "num_indptr must be a whole number"
  )
  expect_error(
    .Call(FLC_DatasetCreateFromCSC_R, indptr, 0L, 0.5, 3L, 3e9, 1L, "verbose=-1", NULL)
    , "nelem must be a whole number"
  )
})

test_that("a sparse row longer than the int32 range is refused before its data is read", {
  model <- .bounds_model()
  handle <- model$.__enclos_env__$private$handle
  # a compact sequence: 3e9 elements without allocating them
  big <- 1.0:3e9
  expect_error(
    .Call(FLC_BoosterPredictForCSRSingleRow_R, handle, 0L, big, 2L, FALSE, FALSE, FALSE, 0L, -1L, "", numeric(1L))
    , "A sparse row cannot hold more than"
  )
  fast <- .Call(FLC_BoosterPredictForCSRSingleRowFastInit_R, handle, 2L, FALSE, FALSE, FALSE, 0L, -1L, "")
  expect_error(
    .Call(FLC_BoosterPredictForCSRSingleRowFast_R, fast, 0L, big, numeric(1L))
    , "A sparse row cannot hold more than"
  )
})
