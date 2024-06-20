from main import local_download_dir, allow_patterns_prefix
import os

pattern = "pattern"

pattern_path = local_download_dir + allow_patterns_prefix + pattern + "/"
file_names = [f for f in os.listdir(pattern_path)]
full_paths = [pattern_path + filename for filename in file_names]

print(full_paths)
