// fix
enum { temporal_window = 3 };	
enum { sight_diameter = 16 };
enum { world_size = sight_diameter / 2 };
enum { sight_area = sight_diameter * sight_diameter };
enum { max_skills = 2 };
enum { num_move_dirs = 4 };
enum { num_actions = 1 + num_move_dirs + max_skills };
enum { channels = 5 + max_skills };

enum { MinibatchSize = 32 };
enum { num_stats = 4 + max_skills };
enum { OutputCount = num_actions };
enum { ImageSize = sight_area * channels };
enum { window_length = temporal_window + 1 };
enum { ImageChannels = window_length * channels };
enum { StatChannels = window_length * num_stats };
enum { InputDataSize = window_length * ImageSize };

enum { HiddenLayerSize = 256 };
enum { ImageFeatureSize = 32 };
enum { LowLevelImageFeatureSize = 16 };

enum { LowLevelKernelSize = 4 };
enum { KernelSize = 4 };