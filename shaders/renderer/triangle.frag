#version 450

// input
layout (location = 0) in vec2 texCoords;

// output
layout (location = 0) out vec4 outFragColor;

// buffers/images
layout (set = 0, binding = 0) uniform sampler2DArray samplerArray;
layout (set = 0, binding = 1) uniform ParamsBuffer {
	uint temporal_sample_count;
	uint _padding_0;
	uint _padding_1;
	uint _padding_2;
} params_buf;


void main()
{
	// Average all the samples for this pixel
	vec4 color = vec4(0);
	for (uint i = 0; i < params_buf.temporal_sample_count; i++) {
		vec3 coords = vec3(texCoords, i);
		color += vec4(texture(samplerArray, coords).xyz, 1.0f);
	}
	outFragColor = color / params_buf.temporal_sample_count;

	//vec3 coords = vec3(texCoords, 0);
	//outFragColor = vec4(texture(samplerArray, coords).xyz, 1.0f);
}
