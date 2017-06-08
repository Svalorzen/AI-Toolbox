/**
 * @page Logging How to use AIToolbox logging facilities.
 *
 * Since AIToolbox is a library, it leaves the choice of how to log to you. You
 * can use any framework, library or C++ iostream facilities you want in order
 * to log as you prefer.
 *
 * In order to enable logging you must define, *before including any AIToolbox
 * header*, all macros described below.
 *
 * - AI_SEVERITY_DEBUG,
 * - AI_SEVERITY_INFO,
 * - AI_SEVERITY_WARNING,
 * - AI_SEVERITY_ERROR
 *
 * These four macros represent how the AIToolbox library splits log severities.
 * You can define them however you want, even to empty values, or merge two
 * into a single value.
 *
 * - AI_LOGGER(SEV, ARGS)
 *
 * This macro is how the library tries to log. SEV is the severity, represented
 * by one of the four SEVERITY macros described above. Args are the arguments
 * to be printed, joined by the << operator.
 *
 * An example of how the library uses this macros to log is the following:
 *
 * ```
 * AI_LOGGER(AI_SEVERITY_DEBUG, "Logging a value in debug: " << value << "\n");
 * ```
 *
 * How you redefine the macros will transform how the final logging statement
 * reads, so you'll be able to send all messages to whatever sink you prefer.
 *
 * Logs do *not* contain newlines. Logs do *not* contain file/line information,
 * but you can add them through the redefinition of the AI_LOGGER macro.
 *
 * A simple redefinition of the macro to print all messages to `std::cout` with
 * file and line information could be:
 *
 * ```
 * #define AI_LOGGER(SEV, ARGS) std::cout << __FILE__ ":" << __LINE__ << " " << ARGS << '\n';
 * ```
 *
 * Note that it is possible to obtain logs only from certain files by simply
 * defining the above macros before #including the files where logs are needed,
 * and #undefine them before the other files.
 *
 * This setup has been chosen in order to maintain the best possible
 * performances when logging is disabled. This choice allows to avoid
 * evaluating arguments, storing strings and so on if the above macros are
 * undefined when any header that uses them finds them undefined.
 */
#ifndef AI_LOGGER

#undef AI_SEVERITY_DEBUG
#undef AI_SEVERITY_INFO
#undef AI_SEVERITY_WARNING
#undef AI_SEVERITY_ERROR

#define AI_SEVERITY_DEBUG   0
#define AI_SEVERITY_INFO    0
#define AI_SEVERITY_WARNING 0
#define AI_SEVERITY_ERROR   0

// Prevent redefinition errors in case of multiple headers with no logging
// split by files with logging.
#ifndef AITOOLBOX_IMPL_FAKELOGGER_HEADER_FILE
#define AITOOLBOX_IMPL_FAKELOGGER_HEADER_FILE

namespace AIToolbox::Impl {
    struct FakeLogger {
        FakeLogger(int) {}
        template <typename T>
        FakeLogger & operator<<(const T&) { return *this; }
    };
}

#endif

// With this we avoid logging, while still checking for possible syntax errors.
#define AI_LOGGER(SEV, ARGS)                            \
    while (false) {                                     \
        AIToolbox::Impl::FakeLogger(SEV) << ARGS;       \
    }

#endif
