#version 450

// output write
layout (location = 0) out vec4 outFragColor;

layout (set = 0, binding = 0) uniform CameraBuffer {
	vec4 color;
} camera;

void main()
{
	// return red
	outFragColor = vec4(camera.color.xyz, 1.0f);
}
