#version 450

layout (location = 0) out vec2 texCoords;

void main()
{
	// positions for the 2-triangle quad
	const vec3 positions[6] = vec3[6](
		vec3(-1.f, -1.f, 0.0f),
		vec3(-1.f, 1.f, 0.0f),
		vec3(1.f, 1.f, 0.0f),
		vec3(1.f, 1.f, 0.0f),
		vec3(1.f, -1.f, 0.0f),
		vec3(-1.f, -1.f, 0.0f)
	);
	// texture coordinates for the quad
	const vec2 uvCoords[6] = vec2[6](
		vec2(0, 1),
		vec2(0, 0),
		vec2(1, 0),
		vec2(1, 0),
		vec2(1, 1),
		vec2(0, 1)
	);

	// output the position of each vertex
	gl_Position = vec4(0.9f * positions[gl_VertexIndex], 1.0f);
	texCoords = uvCoords[gl_VertexIndex];
}