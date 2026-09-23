import json
import time
import os
from geopy.geocoders import Nominatim

with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Load existing coordinates if file already exists
if os.path.exists("location_coords.json"):
    with open("location_coords.json", "r", encoding="utf-8") as f:
        coords = json.load(f)
else:
    coords = {}

geolocator = Nominatim(user_agent="postcard_project", timeout=10)

locations = set()

for item in data:
    locations.add((item["origin_city"], item["origin_country"]))
    locations.add((item["receiving_city"], item["receiving_country"]))

print("Total locations:", len(locations))
print("Already saved:", len(coords))
print("Remaining:", len(locations) - len(coords))

for city, country in locations:
    key = f"{city}, {country}"

    # Skip already saved locations
    if key in coords:
        continue

    try:
        location = geolocator.geocode(key, timeout=10)

        if location:
            coords[key] = [location.latitude, location.longitude]

            with open("location_coords.json", "w", encoding="utf-8") as f:
                json.dump(coords, f, indent=4, ensure_ascii=False)

            print("Found:", key, coords[key])
        else:
            print("Not found:", key)

        time.sleep(2)

    except Exception as e:
        print("Error:", key, e)
        time.sleep(2)
