import pickle
import numpy as np
with open('features/vision_openface3_500.pkl', 'rb') as f:
    data = pickle.load(f)
split_data = {}
for k, v in data.items():
    # video ID like "03bSnISJMiM_1" -> valid MOSI format
    if "_" in k:
        parts = k.split("_")
        try:
            # check if last part is digit
            int(parts[-1])
            new_k = f"{'_'.join(parts[:-1])}[{parts[-1]}]"
            split_data[new_k] = v
        except:
            pass

with open('features/split_vision_openface3_500.pkl', 'wb') as f:
    pickle.dump(split_data, f)
print("Saved split 500 frames to split_vision_openface3_500.pkl")
