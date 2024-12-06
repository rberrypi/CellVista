#version 430
uniform sampler2D img;
in vec4 texPosOut;

uniform float width;
uniform float height;
uniform bool show_cross;
out vec4 gl_FragColor;
//LUTS are broken
void main(void)
{
	vec2 pos = texPosOut.st;
	gl_FragColor = texture(img, pos);
	if (show_cross)
	{
		float pixel_x = ceil(pos.x*width);
		float pixel_y = ceil(pos.y*height);
		float height_half = ceil(height / 2);
		float width_half = ceil(width / 2);
		if ((pixel_x == width_half) || (pixel_y == height_half))
		{
			gl_FragColor = vec4(0, 0, 0, 0);
		}
	}
}
