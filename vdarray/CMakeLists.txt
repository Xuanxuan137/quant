


# Add the source in project root directory
aux_source_directory(. DIR_VDARRAY_SRCS)

# Add header file include directories
include_directories(
        ${PROJECT_SOURCE_DIR}/vdarray
)

# Add block directories

# Target
add_library(
        vdarray ${DIR_VDARRAY_SRCS}
)
target_link_libraries(
        vdarray
)