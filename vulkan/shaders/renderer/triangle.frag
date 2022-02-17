#version 450

#constant TEMPORAL_SAMPLE_COUNT

// input
layout (location = 0) in vec2 texCoords;

// output
layout (location = 0) out vec4 outFragColor;

// buffers/images
layout (set = 0, binding = 0) uniform sampler2DArray samplerArray;

void main()
{
	// Average all the samples for this pixel
	vec4 color = vec4(0);
	for (uint i = 0; i < TEMPORAL_SAMPLE_COUNT; i++) {
		vec3 coords = vec3(texCoords, i);
		color += vec4(texture(samplerArray, coords).xyz, 1.0f);
	}
	outFragColor = color / TEMPORAL_SAMPLE_COUNT;
}
