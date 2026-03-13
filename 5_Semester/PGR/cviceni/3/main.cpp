#include <iostream>
#include <SDL3/SDL.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <stb_image.h>

#include <geGL/StaticCalls.h>
#include <geGL/geGL.h>

using namespace ge::gl;

#ifndef RESOURCE_DIR
#define RESOURCE_DIR "."
#endif //RESOURCE_DIR

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
        #version 430
        #line 71

		out vec2 vCoord;

		uniform mat4 view = mat4(1);
		uniform mat4 proj = mat4(1);

        void main() {
            if (gl_VertexID == 0) {
				vCoord = vec2(0, 0);
                gl_Position = proj * view * vec4(vCoord, 0, 1);
            }
            if (gl_VertexID == 1) {
				vCoord = vec2(1, 0);
                gl_Position = proj * view * vec4(vCoord, 0, 1);
            }
            if (gl_VertexID == 2) {
				vCoord = vec2(0, 1);
                gl_Position = proj * view * vec4(vCoord, 0, 1);
            }
            if (gl_VertexID == 3) {
				vCoord = vec2(1, 1);
                gl_Position = proj * view * vec4(vCoord, 0, 1);
            }
        }
    ).";

	auto const fsSrc = R".(
		#version 430
		#line 100

		in vec2 vCoord;

		layout(location=0) out vec4 fColor;
		layout(binding =0) uniform sampler2D tex;

		void main() {
			fColor = texture(tex, vCoord);
      
		}
	 ).";

	auto vs = createShader(GL_VERTEX_SHADER, vsSrc);
	auto fs = createShader(GL_FRAGMENT_SHADER, fsSrc);


	GLuint prg = glCreateProgram();
	glAttachShader(prg, vs);
	glAttachShader(prg, fs);
	glLinkProgram(prg);

	auto viewL = glGetUniformLocation(prg, "view");
	auto projL = glGetUniformLocation(prg, "proj");

	glm::vec3 cameraPosition = glm::vec3(0.5f, 0.5f, 0.7f);
	float YAngle = 0.f;
	float XAngle = 0.f;
	float sensitivity = 0.001f;

	int width, height, nofChannels;
	stbi_set_flip_vertically_on_load(true);
	auto data = stbi_load(RESOURCE_DIR "parrot.jpg", &width, &height, &nofChannels, 0);
	if (!data) {
		std::cerr << "parrot not found!" << std::endl;
	}

	std::cerr << RESOURCE_DIR << std::endl;



	GLuint texture;
	glCreateTextures(GL_TEXTURE_2D, 1, &texture);

	glTextureStorage2D(texture, 1, GL_RGBA8, width, height); // 1 mipmap level, internal format

	// Ensure row alignment is correct (especially for RGB)
	//glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	GLenum format = (nofChannels == 4) ? GL_RGBA : GL_RGB; // Handle RGB vs RGBA
	glTextureSubImage2D(
		texture,
		0, // mip level
		0, 0,
		width, height,
		format,
		GL_UNSIGNED_BYTE,
		data
	);

	stbi_image_free(data);

	//glParameteri - disable mipmap / set nearest filtering
	glTextureParameteri(texture, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTextureParameteri(texture, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTextureParameteri(texture, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTextureParameteri(texture, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);


	//glPixelStorei - pro urceni velikosti radku obrazku

	//glEnable(GL_DEPTH_TEST);

	bool running = true;
	while (running) { //main loop
		SDL_Event event;

		glm::mat4 viewR = glm::rotate(glm::mat4(1.0f), YAngle, glm::vec3(0.f, 1.f, 0.f)); // Rx
		viewR = glm::rotate(viewR, XAngle, glm::vec3(1.f, 0.f, 0.f)); // Rx * Ry


		while (SDL_PollEvent(&event)) { // event loop
			if (event.type == SDL_EVENT_QUIT) {
				running = false;
			}
			if (event.type == SDL_EVENT_KEY_DOWN) {
				cameraPosition += glm::vec3(glm::transpose(viewR)[2]) * 0.1f * (float)((event.key.key == SDLK_S) - (event.key.key == SDLK_W));
				cameraPosition += glm::vec3(glm::transpose(viewR)[1]) * 0.1f * (float)((event.key.key == SDLK_SPACE) - (event.key.key == SDLK_LSHIFT));
				cameraPosition += glm::vec3(glm::transpose(viewR)[0]) * 0.1f * (float)((event.key.key == SDLK_D) - (event.key.key == SDLK_A));
			}
			if (event.type == SDL_EVENT_MOUSE_MOTION) {
				if (event.motion.state & SDL_BUTTON_LEFT) {
					XAngle += event.motion.yrel * sensitivity;
					YAngle += event.motion.xrel * sensitivity;
				}
			}
		}
		//draw here
		glClearColor(0.1, 0.1, 0.1, 1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glUseProgram(prg);

		// Projection matrix
		auto proj = glm::perspective(glm::half_pi<float>(), 1024.0f / 768.0f, 0.1f, 1000.0f); // proj = glm::perspecive
		glProgramUniformMatrix4fv(prg, projL, 1, GL_FALSE, (float*)&proj);

		// View matrix 
		glm::mat4 view = viewR * glm::translate(glm::mat4(1.0f), -cameraPosition); // view = viewT * viewR
		glProgramUniformMatrix4fv(prg, viewL, 1, GL_FALSE, (float*)&view);

		// Texture Bind
		glBindTextureUnit(0, texture);
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

		SDL_GL_SwapWindow(window);
	}
	SDL_GL_DestroyContext(context);
	SDL_DestroyWindow(window);

	return 0;
}