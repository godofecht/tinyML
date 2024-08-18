#ifndef LOGGER_H
#define LOGGER_H

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>
#include <ctime>

namespace LoggerNS
{
    class Logger
    {
    public:
        enum class VerbosityLevel
        {
            NONE,
            ERROR,
            INFO,
            DEBUG
        };

        Logger (VerbosityLevel verbosity = VerbosityLevel::INFO) : verbosity_(verbosity) {}

        void logInfo (const std::string& message)
        {
            if (verbosity_ >= VerbosityLevel::INFO)
            {
                log ("INFO", message);
            }
        }

        void logError (const std::string& message)
        {
            if (verbosity_ >= VerbosityLevel::ERROR)
            {
                log ("ERROR", message);
            }
        }

        void logDebug (const std::string& message)
        {
            if (verbosity_ >= VerbosityLevel::DEBUG)
            {
                log ("DEBUG", message);
            }
        }

        void logResult (const std::string& label, const std::vector<double>& expected, const std::vector<double>& actual)
        {
            if (verbosity_ >= VerbosityLevel::INFO)
            {
                std::ostringstream oss;
                oss << label << " - Expected: [";
                for (size_t i = 0; i < expected.size(); ++i)
                {
                    oss << std::fixed << std::setprecision(4) << expected[i];
                    if (i < expected.size() - 1)
                        oss << ", ";
                }
                oss << "], Actual: [";
                for (size_t i = 0; i < actual.size(); ++i)
                {
                    oss << std::fixed << std::setprecision(4) << actual[i];
                    if (i < actual.size() - 1)
                        oss << ", ";
                }
                oss << "]";
                logInfo (oss.str());
            }
        }

        void setVerbosity(VerbosityLevel verbosity)
        {
            verbosity_ = verbosity;
        }

    private:
        VerbosityLevel verbosity_;

        void log (const std::string& level, const std::string& message)
        {
            std::ostringstream oss;
            oss << "[" << getCurrentTime() << "] " << level << ": " << message;
            std::cout << oss.str() << std::endl;
        }

        std::string getCurrentTime()
        {
            std::time_t now;
            std::time(&now);

            char buffer[100];
            std::tm timeinfo;

#ifdef _WIN32
            // Use localtime_s on Windows (thread-safe)
            localtime_s(&timeinfo, &now);
#else
            // Use localtime_r on POSIX systems (thread-safe)
            localtime_r(&now, &timeinfo);
#endif

            std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &timeinfo);
            return std::string(buffer);
        }
    };
}

#endif // LOGGER_H
