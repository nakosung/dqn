// fix
enum { temporal_window = 3 };	
enum { sight_diameter = 40 };
enum { world_size = sight_diameter / 2 };
enum { sight_area = sight_diameter * sight_diameter };
enum { num_actions = 6 };

enum { MinibatchSize = 32 };
enum { InputDataSize = sight_area * (temporal_window + 1) };
enum { OutputCount = num_actions };
enum { ImageSize = sight_area };
