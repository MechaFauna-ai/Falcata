test_that("fast prediction refuses cleared external pointers", {
  for (empty in list(NULL, new("externalptr"))) {
  expect_error(
    .Call(FLC_BoosterPredictForMatSingleRowFast_R, empty, c(1.0, 2.0), numeric(1L))
    , "FastConfig handle is null"
  )
  expect_error(
    .Call(FLC_BoosterPredictForCSRSingleRowFast_R, empty, 0L, 1.0, numeric(1L))
    , "FastConfig handle is null"
  )
  }
})
