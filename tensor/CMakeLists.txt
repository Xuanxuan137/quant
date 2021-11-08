


# Add the source in project root directory
aux_source_directory(. DIR_TENSOR_SRCS)

# Add header file include directories
include_directories(
        ${PROJECT_SOURCE_DIR}/tensor
)

# Add block directories

# Target
add_library(
        tensor ${DIR_TENSOR_SRCS}
)
target_link_libraries(
        tensor pthread
)