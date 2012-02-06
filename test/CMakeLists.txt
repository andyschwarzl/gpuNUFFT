#setup testframework
MESSAGE("------setting up testframework-------")

set(GTEST_DIR "../../../gtest-1.6.0")
MESSAGE("GTEST_DIR " ${GTEST_DIR})

include_directories("${GTEST_DIR}/include")

find_library(GTEST_LIB gtest "${GTEST_DIR}/mybuild/Debug")
find_library(GTESTMAIN_LIB gtest_main "${GTEST_DIR}/mybuild/Debug")

#add source dir
add_executable(runUnitTests gridding3D_tests.cc ../src/griddingFunctions.cc)

target_link_libraries(runUnitTests ${GTEST_LIB} ${GTESTMAIN_LIB})
