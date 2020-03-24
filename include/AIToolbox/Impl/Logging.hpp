/**
 * @page logging How Logging Works
 *
 * Since AIToolbox is a library, it leaves the choice of how to log to you. You
 * can use any framework, library or C++ iostream facilities you want.
 *
 * ## Enabling Logging in the Library ##
 *
 * In order to enable logging the library MUST be compiled with the flag
 * AI_LOGGING_ENABLED set to 1 (You can do this via CMake). If it's not, no
 * logging will be produced by the pre-compiled parts of the library.
 *
 * In addition, you need to set the AI_LOGGING_ENABLED macro to 1 before
 * including any AIToolbox header files in your project too, or logging in
 * header-only classes will not work.
 *
 * This has been done in order to obtain the best possible performance when
 * logging is disabled.
 *
 * ## Setting the Logging Function ##
 *
 * In order to log you need to overwrite the AIToolbox::AILogger function
 * pointer. This is a pointer to a function with signature:
 *
 *     void(int severity, const char * message);
 *
 * So for example, in your code you could define:
 *
 *     void myLoggerFunction(int severity, const char * message) {
 *         std::cout << '[' << severity << "] " << message << '\n';
 *     }
 *
 * And then, at the beginning of your main function:
 *
 *     AIToolbox::AILogger = myLoggerFunction;
 *
 * And this would start logging automatically with no filtering. Remember to
 * include the logging header!
 *
 *     #include <AIToolbox/Impl/Logging.hpp>
 *
 * ## Priorities and Log Information ##
 *
 * AIToolbox defines 4 different severity levels:
 *
 * - AI_SEVERITY_DEBUG    (0),
 * - AI_SEVERITY_INFO     (1),
 * - AI_SEVERITY_WARNING  (2),
 * - AI_SEVERITY_ERROR    (3)
 *
 * These values represent how the AIToolbox library splits log severities.
 *
 * Logs do *not* contain newlines. Logs do *not* contain file/line information.
 *
 * The max length of logs is capped at compile time. The variable `logBuffer`
 * is a char array which temporarily contains the message before it is passed
 * to your function. By default, the length of this array is 500. If you need
 * longer logs, feel free to change the length of the array and recompile.
 */

#ifndef AI_TOOLBOX_IMPL_LOGGING_HEADER_FILE
#define AI_TOOLBOX_IMPL_LOGGING_HEADER_FILE

#ifndef AI_LOGGING_ENABLED
#define AI_LOGGING_ENABLED 0
#endif

#define AI_SEVERITY_DEBUG   0
#define AI_SEVERITY_INFO    1
#define AI_SEVERITY_WARNING 2
#define AI_SEVERITY_ERROR   3

#if AI_LOGGING_ENABLED == 1

#include <sstream>

namespace AIToolbox {
    using AILoggerFun = void(int, const char *);

    /**
     * @brief This pointer defines the function used to log.
     *
     * \sa \ref Logging
     */
    inline AILoggerFun * AILogger = nullptr;

    namespace Impl {
        // We use this to dump logs in.
        inline char logBuffer[500] = {0};
    }
}

#else
// Fake logger to keep static checks
namespace AIToolbox::Impl {
    struct FakeLogger {
        FakeLogger(int) {}
        template <typename T>
        FakeLogger & operator<<(const T&) { return *this; }
    };
}

#endif

#if AI_LOGGING_ENABLED == 1
// Actual logging if logging is enabled.
#define AI_LOGGER(SEV, ARGS)                                \
    do {                                                    \
        if (AILogger) {                                     \
            std::stringstream internal_stringstream_;       \
            internal_stringstream_.rdbuf()->pubsetbuf(      \
                AIToolbox::Impl::logBuffer,                 \
                sizeof(AIToolbox::Impl::logBuffer) - 1      \
            );                                              \
            internal_stringstream_ << ARGS << '\0';         \
            AILogger(SEV, AIToolbox::Impl::logBuffer);      \
        }                                                   \
    } while(0)

#else
// Statement to enable static checks on inputs if logging is disabled.
#define AI_LOGGER(SEV, ARGS)                                \
    while (0) {                                             \
        AIToolbox::Impl::FakeLogger(SEV) << ARGS;           \
    }
#endif

#endif
