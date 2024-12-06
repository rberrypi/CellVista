#version 430
uniform sampler2D img;
uniform sampler1D img_lut;
uniform sampler2D img_transformed;
uniform sampler1D img_transformed_lut;
in vec4 texPosOut;

uniform float width;
uniform float height;
uniform bool show_cross;
out vec4 gl_FragColor;

void main(void)
{
	vec2 pos = texPosOut.st;

	vec4 img_rgb = texture(img_lut, texture(img, pos).r);
	vec4 img_transformed_rgb = texture(img_transformed_lut, texture(img_transformed, pos).r);
	float alpha = 0.5;
	gl_FragColor = 2.0 * (alpha * img_rgb + (1.0 - alpha) * img_transformed_rgb);
	//gl_FragColor = img_transformed_rgb;
	if (show_cross)
	{
		float width_half = ceil(width / 2);
		float pixel_x = ceil(pos.x * width);
		float pixel_y = ceil(pos.y * height);
		float height_half = ceil(height / 2);

		if ((pixel_x == width_half) || (pixel_y == height_half))
		{
			gl_FragColor = texture(img_lut, 0.0);
		}
	}
}
