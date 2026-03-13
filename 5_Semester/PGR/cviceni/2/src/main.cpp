#include<iostream>
#include<SDL3/SDL.h>
#include<geGL/StaticCalls.h>
#include <geGL/geGL.h>

using namespace ge::gl;

std::string shaderToStr(GLenum type) {
	if (type == GL_VERTEX_SHADER) {
		return "GL_VERTEX_SHADER";
	}
	if (type == GL_FRAGMENT_SHADER) {
		return "GL_FRAGMENT_SHADER";
	}
	if (type == GL_GEOMETRY_SHADER) {
		return "GL_GEOMETRY_SHADER";
	}
	if (type == GL_TESS_CONTROL_SHADER) {
		return "GL_TESS_CONTROL_SHADER";
	}
	if (type == GL_TESS_EVALUATION_SHADER) {
		return "GL_TESS_EVALUATION_SHADER";
	}
	if (type == GL_COMPUTE_SHADER) {
		return "GL_COMPUTE_SHADER";
	}
	return "";
}

GLuint createShader(GLenum type, std::string const& src) {
	char const* srcs[] = {
		src.c_str()
	};
	GLuint s = glCreateShader(type);
	glShaderSource(s, 1, srcs, nullptr);
	glCompileShader(s);

	GLint status;
	glGetShaderiv(s, GL_COMPILE_STATUS, &status);
	if (status != GL_TRUE) {
		char buffer[1024] = { 0 };
		glGetShaderInfoLog(s, 1024, nullptr, buffer);
		std::cerr << "ERROR: " << shaderToStr(type) << " shader compilation failed" << std::endl;
		std::cerr << std::string(buffer) << std::endl;
	}
	return s;
}

int main(int argc, char* argv[]) {
	std::cerr << "papousek blablabla" << std::endl;

	auto window = SDL_CreateWindow("PGR", 1024, 768, SDL_WINDOW_OPENGL);
	auto context = SDL_GL_CreateContext(window);

	ge::gl::init();



    // Vertex Shader Source
    auto const vsSrc = R".(
        #version 330
        #line 62

        out vec3 vColor;

        void main() {
            if (gl_VertexID == 0) {
                vColor = vec3(1, 0, 0); // Red color
                gl_Position = vec4(-0.5, -0.5, 0, 1);  // Bottom-left vertex
            }
            if (gl_VertexID == 1) {
                vColor = vec3(0, 1, 0); // Green color
                gl_Position = vec4(0.5, -0.5, 0, 1);  // Bottom-right vertex
            }
            if (gl_VertexID == 2) {
                vColor = vec3(0, 0, 1); // Blue color
                gl_Position = vec4(0, 0.5, 0, 1);  // Top vertex
            }
            if (gl_VertexID == 3) {
                vColor = vec3(1, 1, 1); // White color for the dot
                gl_Position = vec4(0.75, 0.25, 0, 1);  // Position for the dot to the right of the triangle
            }
            if (gl_VertexID == 4) {
                vColor = vec3(1, 1, 1); // White color for the dot
                gl_Position = vec4(-0.75, 0.25, 0, 1);  // Position for the dot to the right of the triangle
            }
            if (gl_VertexID == 5) {
                vColor = vec3(1, 1, 1); // White color for the dot
                gl_Position = vec4(0, -0.75, 0, 1);  // Position for the dot to the right of the triangle
            }
        }
    ).";

	auto const fsSrc = R".(
		#version 330
		#line 88

		in vec3 vColor;
		out vec4 fColor;

		void main() {
			fColor = vec4(vColor,1);
      
		}
	 ).";

	auto vs = createShader(GL_VERTEX_SHADER, vsSrc);
	auto fs = createShader(GL_FRAGMENT_SHADER, fsSrc);


	GLuint prg = glCreateProgram();
	glAttachShader(prg, vs);
	glAttachShader(prg, fs);
	glLinkProgram(prg);



	bool running = true;
	while (running) { //main loop
		SDL_Event event;
		while (SDL_PollEvent(&event)) { // event loop
			if (event.type == SDL_EVENT_QUIT) {
				running = false;
			}
		}
		//draw here
		glClearColor(0.1, 0.1, 0.1, 1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glUseProgram(prg);
		glDrawArrays(GL_TRIANGLES, 0, 3);
		glDrawArrays(GL_TRIANGLES, 1, 3);
		glPointSize(30.0f);  // Set the point size to 30 pixels diameter
		glDrawArrays(GL_POINTS, 0, 4);
		SDL_GL_SwapWindow(window);
	}
	SDL_GL_DestroyContext(context);
	SDL_DestroyWindow(window);
	return 0;
}