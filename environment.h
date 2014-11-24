struct Environment
{
	mutable std::mt19937 random_engine;

	bool test_prob(float prob) const 
	{
		const float dice = std::uniform_real_distribution<float>(0,1)(random_engine);		
		return dice < prob;
	}

	int randint(int N) const
	{
		return std::uniform_int_distribution<>(0,N-1)(random_engine);
	}
};