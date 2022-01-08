#version 450

// input
layout (location = 0) in vec2 texCoords;

// output
layout (location = 0) out vec4 outFragColor;

// buffers/images
layout (set = 0, binding = 0) uniform sampler2D tex1;


void main()
{
	outFragColor = vec4(texture(tex1, texCoords).xyz, 1.0f);
	//outFragColor = vec4(camera.color.xyz, 1.0f);
}
