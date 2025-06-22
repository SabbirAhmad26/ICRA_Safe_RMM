import json

with open("./map06.json") as f:
	config = json.load(f)

print(config['highway-6']['Base_locations'])
print(config['highway-6']['Random_loc'])
print(config['highway-6']['Target_speeds'])
print(config['highway-6']['Destination'])