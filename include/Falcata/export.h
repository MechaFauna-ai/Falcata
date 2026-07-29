/*!
 * Copyright (c) 2017-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2017-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef FALCATA_INCLUDE_FALCATA_EXPORT_H_
#define FALCATA_INCLUDE_FALCATA_EXPORT_H_

/** Macros for exporting symbols in MSVC/GCC/CLANG **/

#ifdef __cplusplus
#define FALCATA_EXTERN_C extern "C"
#else
#define FALCATA_EXTERN_C
#endif


#ifdef _MSC_VER
#define FALCATA_EXPORT __declspec(dllexport)
#define FALCATA_C_EXPORT FALCATA_EXTERN_C __declspec(dllexport)
#else
#define FALCATA_EXPORT
#define FALCATA_C_EXPORT FALCATA_EXTERN_C
#endif

#endif  // FALCATA_INCLUDE_FALCATA_EXPORT_H_
