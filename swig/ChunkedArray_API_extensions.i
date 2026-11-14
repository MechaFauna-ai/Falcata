/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
/**
 * Wrap chunked_array.hpp class for SWIG usage.
 *
 * Author: Alberto Ferreira
 */

%{
#include "../include/Falcata/utils/chunked_array.hpp"
%}

// ChunkedArray reports errors with Log::Fatal (a C++ exception): chunk size 0, a failed chunk allocation, ...
// Unhandled, the exception crossed the JNI boundary and called std::terminate, which killed the JVM. Turn it into
// a java.lang.RuntimeException instead.
%exception {
  try {
    $action
  } catch (const std::exception& e) {
    SWIG_JavaThrowException(jenv, SWIG_JavaRuntimeException, e.what());
    return $null;
  }
}

%include "../include/Falcata/utils/chunked_array.hpp"

using Falcata::ChunkedArray;

%template(int32ChunkedArray) ChunkedArray<int32_t>;
/* Unfortunately, for the time being,
 * SWIG has issues generating the overloads to coalesce_to()
 * for larger integral types
 * so we won't support that for now:
 */
//%template(int64ChunkedArray) ChunkedArray<int64_t>;
%template(floatChunkedArray) ChunkedArray<float>;
%template(doubleChunkedArray) ChunkedArray<double>;

%exception;
