#version 450

// input
layout (location = 0) in vec2 texCoords;

// output
layout (location = 0) out vec4 outFragColor;

// buffers/images
layout (set = 0, binding = 0) uniform sampler2DArray samplerArray;


void main()
{
	vec3 coords = vec3(texCoords, 0);
	outFragColor = vec4(texture(samplerArray, coords).xyz, 1.0f);
}
