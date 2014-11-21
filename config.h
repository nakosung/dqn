// fix
enum { temporal_window = 3 };	
enum { sight_diameter = 32 };
enum { world_size = sight_diameter / 2 };
enum { num_states = sight_diameter * sight_diameter };
enum { num_actions = 6 };

enum { MinibatchSize = 32 };
enum { InputDataSize = num_states * (temporal_window + 1) };
enum { OutputCount = num_actions };
enum { ImageSize = num_states };
