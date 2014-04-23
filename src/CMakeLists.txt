MESSAGE("------start creating libraries-------")

SET(GRIDDING_SRC_DIR ${CMAKE_SOURCE_DIR}/src)
SET(GRIDDING_SOURCES ${GRIDDING_SRC_DIR}/gridding_utils.cpp
                     ${GRIDDING_SRC_DIR}/gridding_operator_factory.cpp
                     ${GRIDDING_SRC_DIR}/gridding_operator.cpp
										 ${GRIDDING_SRC_DIR}/texture_gridding_operator.cpp
										 ${GRIDDING_SRC_DIR}/balanced_gridding_operator.cpp)

ADD_SUBDIRECTORY(gpu)

#cpu not necessary
#ADD_SUBDIRECTORY(cpu)
