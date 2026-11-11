test_that("prediction counts cannot wrap R integer buffers", {
  model <- lgb.train(
    params = list(objective = "regression", min_data_in_leaf = 1L, num_threads = 1L, verbosity = -1L)
    , data = lgb.Dataset(matrix(seq_len(80L), ncol = 2L), label = seq_len(40L))
    , nrounds = 2L
  )
  handle <- model$.__enclos_env__$private$handle
  expect_error(
    .Call(FLC_BoosterCalcNumPredict_R, handle, .Machine$integer.max, 0L, 1L, 0L, 0L, -1L, integer(1L))
    , "Prediction length exceeds the R integer limit"
  )
})
