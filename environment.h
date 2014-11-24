struct Environment
{
	Environment(std::mt19937& random_engine)
	: random_engine(random_engine)
	{}

	std::mt19937& random_engine;

	bool test_prob(float prob)
	{
		const float dice = std::uniform_real_distribution<float>(0,1)(random_engine);		
		return dice < prob;
	}

	int randint(int N)
	{
		return std::uniform_int_distribution<>(0,N-1)(random_engine);
	}
};