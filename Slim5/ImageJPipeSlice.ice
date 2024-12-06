//"C:\Program Files (x86)\ZeroC\Ice-3.6.3\bin\slice2cpp.exe" --output-dir="C:\fslim2\Slim5" "C:\fslim2\Slim5\ImageJPipeSlice.ice"
#pragma once
module QLI
{
	["cpp:type:wstring"]
	struct ImageJImageMetaData
	{
		string name;
		string lutName;
		float pixelratio;
		float displaymin;
		float displaymax;
	};
	struct segmentation_settings
	{
		 float areamin;
		 float areamax;
		 float circmin;
		 float circmax;
		 bool islive;
	};
	sequence<float> ImageJInteropBuffer;
	sequence<byte> ImageJInteropSegementation;
	interface ImageJInterop
	{
		void sendImage(int width, int height, ["cpp:array"] ImageJInteropBuffer input, ImageJImageMetaData metadata);
		void sendSegmentation(int width, int height, ["cpp:array"] ImageJInteropBuffer massmap, ["cpp:array"] ImageJInteropSegementation binarymap, ImageJImageMetaData metadata,segmentation_settings segmentationmetadata);
	};
};