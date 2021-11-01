


# Add the source in project root directory
aux_source_directory(. DIR_NN_SRCS)

# Add header file include directories
include_directories(
        ${PROJECT_SOURCE_DIR}/nn
        ${PROJECT_SOURCE_DIR}/vdarray
        ${PROJECT_SOURCE_DIR}/util
)

# Add block directories

# Target
add_library(
        nn ${DIR_NN_SRCS}
)
target_link_libraries(
        nn
)