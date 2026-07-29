#' @name setLGBMThreads
#' @title Set maximum number of threads used by Falcata
#' @description Falcata attempts to speed up many operations by using multi-threading.
#'              The number of threads used in those operations can be controlled via the
#'              \code{num_threads} parameter passed through \code{params} to functions like
#'              \link{lgb.train} and \link{lgb.Dataset}. However, some operations (like materializing
#'              a model from a text file) are done via code paths that don't explicitly accept thread-control
#'              configuration.
#'
#'              Use this function to set the maximum number of threads Falcata will use for such operations.
#'
#'              This function affects all Falcata operations in the same process.
#'
#'              So, for example, if you call \code{setFalcataThreads(4)}, no other multi-threaded Falcata
#'              operation in the same process will use more than 4 threads.
#'
#'              Call \code{setFalcataThreads(-1)} to remove this limitation.
#' @param num_threads maximum number of threads to be used by Falcata in multi-threaded operations
#' @return NULL
#' @seealso \link{getFalcataThreads}
#' @export
setFalcataThreads <- function(num_threads) {
    .Call(
        FLC_SetMaxThreads_R,
        num_threads
    )
    return(invisible(NULL))
}

#' @name getLGBMThreads
#' @title Get default number of threads used by Falcata
#' @description Falcata attempts to speed up many operations by using multi-threading.
#'              The number of threads used in those operations can be controlled via the
#'              \code{num_threads} parameter passed through \code{params} to functions like
#'              \link{lgb.train} and \link{lgb.Dataset}. However, some operations (like materializing
#'              a model from a text file) are done via code paths that don't explicitly accept thread-control
#'              configuration.
#'
#'              Use this function to see the default number of threads Falcata will use for such operations.
#' @return number of threads as an integer. \code{-1} means that in situations where parameter \code{num_threads} is
#'         not explicitly supplied, Falcata will choose a number of threads to use automatically.
#' @seealso \link{setFalcataThreads}
#' @export
getFalcataThreads <- function() {
    out <- 0L
    .Call(
        FLC_GetMaxThreads_R,
        out
    )
    return(out)
}

#' @name setLGBMthreads
#' @title Deprecated alias for \code{setFalcataThreads}
#' @description Kept for code written against the pre-rename name.
#' @param num_threads number of threads to use
#' @return NULL
#' @export
setLGBMthreads <- function(num_threads) {
  setFalcataThreads(num_threads)
}

#' @name getLGBMthreads
#' @title Deprecated alias for \code{getFalcataThreads}
#' @description Kept for code written against the pre-rename name.
#' @return number of threads configured for Falcata
#' @export
getLGBMthreads <- function() {
  getFalcataThreads()
}
