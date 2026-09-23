import json
import time
from geopy.geocoders import Nominatim

with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

countries = set()

for item in data:
    countries.add(item.get("origin_country", ""))
    countries.add(item.get("receiving_country", ""))

countries.discard("")

geolocator = Nominatim(user_agent="postcard_country_centers", timeout=10)

country_centers = {}

for country in sorted(countries):
    try:
        location = geolocator.geocode(country)

        if location:
            country_centers[country] = [location.latitude, location.longitude]
            print("Found:", country)

        time.sleep(1)

    except Exception as e:
        print("Error:", country, e)

with open("country_centers.json", "w", encoding="utf-8") as f:
    json.dump(country_centers, f, indent=4, ensure_ascii=False)
