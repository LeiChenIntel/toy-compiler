#ifndef UTILS_UTILITY_H
#define UTILS_UTILITY_H

enum class Platform { Device1 = 0, Device2 };

Platform getPlatformFromCmd(int argc, char *argv[]);

#endif // UTILS_UTILITY_H
