MESSAGE("------start creating libraries-------")

SET(GRIDDING_SRC_DIR ${CMAKE_SOURCE_DIR}/src)
SET(GRIDDING_SOURCES ${GRIDDING_SRC_DIR}/griddingFunctions.cpp
                     ${GRIDDING_SRC_DIR}/gridding_operator_factory.cpp
										 ${GRIDDING_SRC_DIR}/gridding_operator_matlabfactory.cpp
                     ${GRIDDING_SRC_DIR}/gridding_operator.cpp)

ADD_SUBDIRECTORY(gpu)

#cpu not necessary
#ADD_SUBDIRECTORY(cpu)
