cmake_minimum_required(VERSION 3.19)

project(PGR)

set_property(
	DIRECTORY 
	${CMAKE_CURRENT_SOURCE_DIR} 
	PROPERTY 
	VS_STARTUP_PROJECT 
	${PROJECT_NAME}
)

option(SDL_SHARED "" OFF)
option(SDL_STATIC "" ON)

add_subdirectory(libs/SDL-release-3.2.24)
add_subdirectory(libs/geGL-master)
add_subdirectory(libs/glm-1.0.2)

add_executable(${PROJECT_NAME} 
	src/main.cpp
	libs/stb_image/stb_image.h
	libs/stb_image/impl.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC 
	libs/stb_image
)

target_compile_definitions(${PROJECT_NAME} PUBLIC
	RESOURCE_DIR="${CMAKE_SOURCE_DIR}/resources/"
)

target_link_libraries(
	${PROJECT_NAME}
	PUBLIC 
	geGL::geGL
	SDL3::SDL3
	glm::glm
)