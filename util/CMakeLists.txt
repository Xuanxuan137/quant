

# Add the source in project root directory
aux_source_directory(. DIR_UTIL_SRCS)

# Add header file include directories
include_directories(
        ${PROJECT_SOURCE_DIR}/nn
        ${PROJECT_SOURCE_DIR}/tensor
        ${PROJECT_SOURCE_DIR}/util
)

# Add block directories

# Target
add_library(
        util ${DIR_UTIL_SRCS}
)
target_link_libraries(
        util tensor
)