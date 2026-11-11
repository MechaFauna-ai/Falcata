/*!
 * Copyright (c) 2016-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef FALCATA_INCLUDE_FALCATA_UTILS_LOG_H_
#define FALCATA_INCLUDE_FALCATA_UTILS_LOG_H_

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>

#ifdef LGB_R_BUILD

#ifndef R_NO_REMAP
#define R_NO_REMAP
#endif

#ifndef R_USE_C99_IN_CXX
#define R_USE_C99_IN_CXX
#endif

#include <R_ext/Print.h>
extern "C" void R_FlushConsole(void);
#endif

namespace Falcata {

#if defined(_MSC_VER)
#define THREAD_LOCAL __declspec(thread)
#else
#define THREAD_LOCAL thread_local
#endif

#ifndef CHECK
// do/while(0) makes the macro a single statement, so CHECK(x); is safe as the
// body of an unbraced if/else (the old bare if bound a following else to it)
#define CHECK(condition)                                                        \
  do {                                                                          \
    if (!(condition))                                                           \
      Log::Fatal("Check failed: " #condition " at %s, line %d .\n", __FILE__,   \
                 __LINE__);                                                     \
  } while (0)
#endif

#ifndef CHECK_EQ
#define CHECK_EQ(a, b) CHECK((a) == (b))
#endif

#ifndef CHECK_NE
#define CHECK_NE(a, b) CHECK((a) != (b))
#endif

#ifndef CHECK_GE
#define CHECK_GE(a, b) CHECK((a) >= (b))
#endif

#ifndef CHECK_LE
#define CHECK_LE(a, b) CHECK((a) <= (b))
#endif

#ifndef CHECK_GT
#define CHECK_GT(a, b) CHECK((a) > (b))
#endif

#ifndef CHECK_LT
#define CHECK_LT(a, b) CHECK((a) < (b))
#endif

#ifndef CHECK_NOTNULL
#define CHECK_NOTNULL(pointer)                                           \
  do {                                                                   \
    if ((pointer) == nullptr)                                            \
      Falcata::Log::Fatal(#pointer " Can't be NULL at %s, line %d .\n", \
                           __FILE__, __LINE__);                          \
  } while (0)
#endif

enum class LogLevel : int {
  Fatal = -1,
  Warning = 0,
  Info = 1,
  Debug = 2,
};

/*!
 * \brief A static Log class
 */
class Log {
 public:
  using Callback = void (*)(const char *);
  /*!
   * \brief Resets the minimal log level. It is INFO by default.
   * \param level The new minimal log level.
   */
  static void ResetLogLevel(LogLevel level) {
    ThreadSetting().level = level;
    ThreadSetting().level_set = true;
    DefaultLevel().store(level, std::memory_order_relaxed);
  }

  static void ResetCallBack(Callback callback) {
    ThreadSetting().callback = callback;
    ThreadSetting().callback_set = true;
    DefaultCallBack().store(callback, std::memory_order_relaxed);
  }

  /*! \brief The calling thread's log level and callback: a thread keeps what it set itself with ResetLogLevel /
   *  ResetCallBack, and a thread that never set them (an OpenMP worker, a thread the library starts) uses the
   *  last ones any thread set */
  static LogLevel CurrentLevel() { return GetLevel(); }
  static Callback CurrentCallBack() { return GetLogCallBack(); }

  static void Debug(const char *format, ...) {
    va_list val;
    va_start(val, format);
    Write(LogLevel::Debug, "Debug", format, val);
    va_end(val);
  }
  static void Info(const char *format, ...) {
    va_list val;
    va_start(val, format);
    Write(LogLevel::Info, "Info", format, val);
    va_end(val);
  }
  static void Warning(const char *format, ...) {
    va_list val;
    va_start(val, format);
    Write(LogLevel::Warning, "Warning", format, val);
    va_end(val);
  }
  static void Fatal(const char *format, ...) {
    va_list val;
    const size_t kBufSize = 1024;
    char str_buf[kBufSize];
    va_start(val, format);
#ifdef _MSC_VER
    vsnprintf_s(str_buf, kBufSize, format, val);
#else
    vsnprintf(str_buf, kBufSize, format, val);
#endif
    va_end(val);

// R code should write back to R's error stream,
// otherwise to stderr
#ifndef LGB_R_BUILD
    fprintf(stderr, "[Falcata] [Fatal] %s\n", str_buf);
    fflush(stderr);
#else
    REprintf("[Falcata] [Fatal] %s\n", str_buf);
    R_FlushConsole();
#endif
    throw std::runtime_error(std::string(str_buf));
  }

 private:
  static void Write(LogLevel level, const char *level_str, const char *format,
                    va_list val) {
    if (level <= GetLevel()) {  // omit the message with low level
      // one message at a time: a message is written in pieces (the callback receives the prefix, the text and the
      // newline separately and the Python side joins them), and the CUDA forest trainer's slot threads log at once
      std::lock_guard<std::mutex> lock(WriteMutex());
// R code should write back to R's output stream,
// otherwise to stdout
#ifndef LGB_R_BUILD
      const Callback callback = GetLogCallBack();
      if (callback == nullptr) {
        printf("[Falcata] [%s] ", level_str);
        vprintf(format, val);
        printf("\n");
        fflush(stdout);
      } else {
        const size_t kBufSize = 512;
        char buf[kBufSize];
        snprintf(buf, kBufSize, "[Falcata] [%s] ", level_str);
        callback(buf);
        vsnprintf(buf, kBufSize, format, val);
        callback(buf);
        callback("\n");
      }
#else
      Rprintf("[Falcata] [%s] ", level_str);
      Rvprintf(format, val);
      Rprintf("\n");
      R_FlushConsole();
#endif
    }
  }

  // a trick to use static variable in header file.
  // May be not good, but avoid to use an additional cpp file
  // a plain aggregate, constant-initialized (as __declspec(thread) requires)
  struct PerThread {
    bool level_set;
    bool callback_set;
    LogLevel level;
    Callback callback;
  };

  static PerThread &ThreadSetting() {
    static THREAD_LOCAL PerThread setting = {false, false, LogLevel::Info, nullptr};
    return setting;
  }

  // the last level / callback set by any thread
  static std::atomic<LogLevel> &DefaultLevel() {
    static std::atomic<LogLevel> level(LogLevel::Info);
    return level;
  }

  static std::atomic<Callback> &DefaultCallBack() {
    static std::atomic<Callback> callback(nullptr);
    return callback;
  }

  static LogLevel GetLevel() {
    const PerThread &setting = ThreadSetting();
    return setting.level_set ? setting.level : DefaultLevel().load(std::memory_order_relaxed);
  }

  static std::mutex &WriteMutex() {
    static std::mutex mutex;
    return mutex;
  }

  static Callback GetLogCallBack() {
    const PerThread &setting = ThreadSetting();
    return setting.callback_set ? setting.callback : DefaultCallBack().load(std::memory_order_relaxed);
  }
};

}  // namespace Falcata
#endif  // FALCATA_INCLUDE_FALCATA_UTILS_LOG_H_
