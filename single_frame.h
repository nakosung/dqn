struct SingleFrame
{
	typedef std::array<float,sight_area> Image;
	std::array<Image,channels> images;
	std::array<float,num_stats> stats;
};