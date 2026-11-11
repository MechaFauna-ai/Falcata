test_that("native model string loader validates inputs and terminates raw vectors", {
  model <- lgb.train(
    params = list(objective = "regression", num_threads = 1L, verbosity = -1L)
    , data = lgb.Dataset(matrix(seq_len(80L), ncol = 2L), label = seq_len(40L))
    , nrounds = 2L
  )
  text <- model$save_model_to_string(NULL)
  expect_type(.Call(FLC_BoosterLoadModelFromString_R, charToRaw(text)), "externalptr")
  expect_error(.Call(FLC_BoosterLoadModelFromString_R, 1L), "raw vector or a string")
  expect_error(.Call(FLC_BoosterLoadModelFromString_R, character()), "exactly one string")
  expect_error(.Call(FLC_BoosterLoadModelFromString_R, c(text, text)), "exactly one string")
})
