struct SingleFrame
{
	typedef std::array<float,ImageSize> Image;
	Image image;

	template <typename T>
	void fill(T& target) const
	{
		std::copy(image.begin(),image.end(),target);
		target += image.size();
	}

	template <typename T>
	static void fill_empty(T& target)
	{
		std::fill(target, target + ImageSize, 0);
		target += ImageSize;
	}

	template <typename T>
	static void fill(T& target, const SingleFrame* frame)
	{
		if (frame)
		{
			frame->fill(target);
		}
		else
		{
			fill_empty(target);
		}
	}

	template <typename T, typename U>
	static void fill_frames(T& target, const U& frames)
	{
		for (const auto& f : frames)
		{	
			SingleFrame::fill(target,f.get());
		}
	}
};