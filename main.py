from flask import Flask, render_template, request
import json
import folium
from folium.plugins import PolyLineTextPath, AntPath
import numpy as np
import torch
import clip
from PIL import Image
import math
import plotly.graph_objects as go
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

with open("location_coords.json","r", encoding="utf-8") as f:
    location_coords = json.load(f)

with open("country_centers.json","r", encoding="utf-8") as f:
    country_centers = json.load(f)

image_names = np.load("names.npy")
image_features=np.load("features.npy")

# Load cluster labels
cluster_labels = np.load("cluster_labels.npy")
 # assigning names for clusters manually
cluster_names={
     0:"Postal Collectibles",
     1:"Historic Architecture",
     2:"Art & Design",
     3:"Urban Lanadscapes",
     4:"Illustrations",
     5:"Cultural Heritages",
     6:"Wildlife Photography",
     7:"Travel Destinations",
     8:"Religious Landmarks",
     9:"Nature",
     10:"Paintings",
     11:"Flowers",
     12:"Urban Architectures",
     13:"African's Lifestyle",
     14:"Scenic Landscape",
     15:"National Flags",
     16:"Buddhism",
     17:"Transportation",
     18:"Ancient Architecture",
     19:"Wild & Marine Life",
     20:"Traditions",
     21:"Beaches",
     22:"Maps",
     23:"Islands",
     24:"Wild Animals",
     25:"Tropical Coastlines",
     26:"Natural Wonders",
     27:"Cultural Artifacts",
     28:"Tribal Communities",
     29:"Island & Coastel Destinations"

 }
topic_hierarchy = {
    "Nature": [
        "Nature",
        "Flowers",
        "Wildlife Photography",
        "Wild Animals",
        "Wild & Marine Life",
        "Beaches",
        "Tropical Coastlines",
        "Natural Wonders",
        "Scenic Landscape"
    ],

    "Architecture": [
        "Historic Architecture",
        "Ancient Architecture",
        "Urban Architectures",
        "Religious Landmarks"
    ],

    "Art & Culture": [
        "Art & Design",
        "Paintings",
        "Illustrations",
        "Cultural Heritages",
        "Cultural Artifacts",
        "Traditions",
        "Tribal Communities",
        "Buddhism"
    ],

    "Travel": [
        "Travel Destinations",
        "Transportation",
        "Islands",
        "Island & Coastel Destinations",
        "Maps"
    ],

    "People": [
        "African's Lifestyle"
    ],

    "Other": [
        "Postal Collectibles",
        "National Flags"
    ]
}

cluster_colors = {
    0: "blue",
    1: "darkred",
    2: "purple",
    3: "cadetblue",
    4: "pink",
    5: "darkgreen",
    6: "green",
    7: "orange",
    8: "black",
    9: "lightgreen",
    10: "beige",
    11: "red",
    12: "darkblue",
    13: "brown",
    14: "lightblue",
    15: "gray",
    16: "purple",
    17: "orange",
    18: "darkred",
    19: "green",
    20: "brown",
    21: "cyan",
    22: "black",
    23: "lightblue",
    24: "orange",
    25: "cadetblue",
    26: "green",
    27: "purple",
    28: "brown",
    29: "blue"
}

# cluster dictionary
cluster_map = {}

for i, name in enumerate(image_names):
    cluster_id = int(cluster_labels[i])
    if cluster_id not in cluster_map:
        cluster_map[cluster_id]=[]
    cluster_map[cluster_id].append(name)

image_to_cluster ={}
for cluster_id, names in cluster_map.items():
    for name in names:
        image_to_cluster[name] = cluster_id


# ---------------- CLIP SEARCH ----------------


topics =[
    "mountains",
    "nature",
    "animals",
    "city",
    "beach",
    "snow",
    "flowers",
    "buildings",
    "forest"
]

# ---------------- HOME ----------------
@app.route("/", methods=["GET", "POST"])


def home():
    page = request.args.get("page", 1, type=int)

    per_image = 42


    query = request.args.get("query","").lower()
    country = request.args.get("country","")
    topic = request.args.get("topic","")
    min_distance = request.args.get("min_km","")
    max_distance = request.args.get("max_km","")

    filtered_data = []
    semantic_results = None
    if query:
        text = clip.tokenize([query]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text)
        text_features/= text_features.norm(dim=-1,keepdim=True)
        similarities=(
            image_features @ text_features.cpu().numpy().T

        ).squeeze()
        top_indices=similarities.argsort()[::-1][:100]
        semantic_results=set(image_names[top_indices])





    distances =[item.get("distance",0) for item in data]
    min_distance_value=min(distances)
    max_distance_value=max(distances)


    for item in data:


        name = item.get("name","")
        if semantic_results is not None:
            if name not in semantic_results:
                continue
            
        origin = item.get("origin_country","")
        destination = item.get("receiving_country","")
        cluster_id = image_to_cluster[item["name"]]
        cluster_topic = cluster_names[cluster_id]
        distance = item.get("distance",0)
        

       

        if country:
            if origin != country and destination != country:
                continue

        if topic:
            selected_topics = []
            if topic in topic_hierarchy:
                selected_topics = topic_hierarchy[topic]
            else:
                selected_topics = [topic]
            if cluster_topic not in selected_topics:
                continue

        if min_distance:
            if float(distance) < float(min_distance):
                continue
          
    
        if max_distance:
            if float(distance) > float(max_distance):
                continue

      
        filtered_data.append(item)        


    


    
    # assuming results already contains all images
    start = (page - 1) * per_image
    end = start + per_image

    current_data= filtered_data[start:end]
    results = []

    for item in current_data:
        print(item)

        travel_time = item.get("time", 0)

        try:
            travel_time = float(travel_time)
        except:
            travel_time = 0

        results.append({
            "name": item.get("name",""),
            "origin": item.get("origin_country",""),
            "destination": item.get("receiving_country",""),
            "origin_city": item.get("origin_city",""),
            "receiving_city": item.get("receiving_city",""),
            "time": travel_time,
            "date_sent": item.get("date_sent",""),
            "date_received": item.get("date_received",""),
            "distance": item.get("distance",0),
            "outlier": travel_time >= 100
        })

    
   
    countries=sorted(set(
            [i.get("origin_country","") for i in data] +
            [i.get("receiving_country","") for i in data]
            ))    
    return render_template(
        "index.html",
        results=results,
        
    
        countries=countries,
    
        query=query,
        page=page,
        total_images=len(filtered_data),
        has_more=end < len(filtered_data),
        min_distance_value=min_distance_value,
        max_distance_value=max_distance_value,
        total_clusters=len(cluster_map),
        cluster_names=cluster_names,
        topic_hierarchy=topic_hierarchy

    )

def curved_line(start, end, curve=0.25, points=80):
    lat1, lon1 = start
    lat2, lon2 = end

    mid_lat = (lat1 + lat2) / 2
    mid_lon = (lon1 + lon2) / 2

    dx = lon2 - lon1
    dy = lat2 - lat1

    control_lat = mid_lat + curve * dx
    control_lon = mid_lon - curve * dy

    coords = []

    for i in range(points + 1):
        t = i / points
        lat = (1 - t) ** 2 * lat1 + 2 * (1 - t) * t * control_lat + t ** 2 * lat2
        lon = (1 - t) ** 2 * lon1 + 2 * (1 - t) * t * control_lon + t ** 2 * lon2
        coords.append([lat, lon])

    return coords


def route_color(count):
    if count >= 20:
        return "darkred"
    elif count >= 10:
        return "orange"
    elif count >= 5:
        return "green"
    elif count >= 2:
        return "blue"
    else:
        return "purple"


def route_weight(count):
   
    return 3


# ---------------- MAP ----------------


@app.route("/map")
def show_map():
    selected_cluster = request.args.get("cluster", type=int)

    route_counts = {}
    country_counts = {}

    for item in data:
        name = item.get("name", "")
        cluster_id = image_to_cluster.get(name)

        if selected_cluster is not None and cluster_id != selected_cluster:
            continue

        origin = item.get("origin_country", "")
        dest = item.get("receiving_country", "")

        if origin in country_centers and dest in country_centers:
            route_key = (origin, dest, cluster_id)

            if route_key not in route_counts:
                route_counts[route_key] = {
                    "start": country_centers[origin],
                    "end": country_centers[dest],
                    "count": 0,
                    "origin": origin,
                    "dest": dest,
                    "cluster_id": cluster_id
                }

            route_counts[route_key]["count"] += 1
            country_counts[origin] = country_counts.get(origin, 0) + 1
            country_counts[dest] = country_counts.get(dest, 0) + 1

    sorted_routes = sorted(
        route_counts.values(),
        key=lambda x: x["count"],
        reverse=True
    )

    if selected_cluster is None:
        overview_routes = sorted_routes[:80]
        medium_routes = []
        detail_routes = []
        title = "Global Travel Overview"
    else:
        overview_routes = sorted_routes [:50]
        medium_routes = []
        detail_routes = []
        title = f"{cluster_names.get(selected_cluster, 'Cluster')} Travel Map"

    fig = go.Figure()

    overview_trace_ids = []
    medium_trace_ids = []
    detail_trace_ids = []

    def add_route(route, level):
        start = route["start"]
        end = route["end"]
        count = route["count"]
        cluster_id = route["cluster_id"]

        curved_coords = curved_line(start, end, curve=0.35, points=40)
        lats = [p[0] for p in curved_coords]
        lons = [p[1] for p in curved_coords]

        if selected_cluster is None:
            color = route_color(count)
        else:
            color = cluster_colors.get(cluster_id, "blue")

        visible = True if level == "overview" else False

        fig.add_trace(go.Scattergeo(
            lon=lons,
            lat=lats,
            mode="lines",
            line=dict(width=2.5, color=color),
            opacity=0.75,
            visible=visible,
            hoverinfo="text",
            text=(
                f"{route['origin']} → {route['dest']}<br>"
                f"Postcards: {count}<br>"
                f"Cluster: {cluster_names.get(cluster_id, 'All')}"
            ),
            showlegend=False
        ))

        trace_id = len(fig.data) - 1

        if level == "overview":
            overview_trace_ids.append(trace_id)
        elif level == "medium":
            medium_trace_ids.append(trace_id)
        else:
            detail_trace_ids.append(trace_id)

    for route in overview_routes:
        add_route(route, "overview")

    for route in medium_routes:
        add_route(route, "medium")

    for route in detail_routes:
        add_route(route, "detail")

    shown_countries = set()
    visible_routes = overview_routes + medium_routes + detail_routes
    

    for route in visible_routes:
        shown_countries.add(route["origin"])
        shown_countries.add(route["dest"])

    country_lats = []
    country_lons = []
    country_text = []
    country_sizes = []

    for country in shown_countries:
        if country not in country_centers:
            continue

        lat, lon = country_centers[country]
        total = country_counts.get(country, 1)

        country_lats.append(lat)
        country_lons.append(lon)
        country_text.append(f"{country}<br>Total postcards: {total}")
        country_sizes.append(max(6, min(math.sqrt(total) * 4, 18)))

    fig.add_trace(go.Scattergeo(
        lon=country_lons,
        lat=country_lats,
        mode="markers",
        marker=dict(
            size=country_sizes,
            color="white",
            line=dict(width=1.5, color="black")
        ),
        hoverinfo="text",
        text=country_text,
        showlegend=False
    ))

    marker_trace_id = len(fig.data) - 1

    fig.update_layout(
        title=dict(
            text=title + "<br><sup>Semantic zoom: zoom in to reveal more postcard routes</sup>",
            x=0.5,
            xanchor="center"
        ),
        height=850,
        margin=dict(l=0, r=0, t=80, b=0),
        geo=dict(
            projection_type="natural earth",
            showland=True,
            landcolor="rgb(245,245,245)",
            showocean=True,
            oceancolor="rgb(220,235,240)",
            showcountries=True,
            countrycolor="rgb(200,200,200)",
            showcoastlines=True,
            coastlinecolor="rgb(180,180,180)"
        )
    )

    html = fig.to_html(
        full_html=True,
        include_plotlyjs=True,
        div_id="postcard_map"
    )

    semantic_zoom_js = f"""
    <script>
    const overviewTraces = {overview_trace_ids};
    const mediumTraces = {medium_trace_ids};
    const detailTraces = {detail_trace_ids};
    const markerTrace = {marker_trace_id};

    const mapDiv = document.getElementById("postcard_map");

    function updateSemanticZoom() {{
        let scale = 1;

        if (
            mapDiv._fullLayout &&
            mapDiv._fullLayout.geo &&
            mapDiv._fullLayout.geo.projection &&
            mapDiv._fullLayout.geo.projection.scale
        ) {{
            scale = mapDiv._fullLayout.geo.projection.scale;
        }}

        let visible = [];

        for (let i = 0; i < mapDiv.data.length; i++) {{
            visible.push(false);
        }}

        overviewTraces.forEach(i => visible[i] = true);

        if (scale >= 1.6) {{
            mediumTraces.forEach(i => visible[i] = true);
        }}

        if (scale >= 2.6) {{
            detailTraces.forEach(i => visible[i] = true);
        }}

        visible[markerTrace] = scale >= 1.6;

        Plotly.restyle(mapDiv, {{visible: visible}});
    }}

    mapDiv.on("plotly_relayout", function() {{
        setTimeout(updateSemanticZoom, 100);
    }});

    setTimeout(updateSemanticZoom, 500);
    </script>
    """

    html = html.replace("</body>", semantic_zoom_js + "</body>")

    return html


#--------------individual poscart journey-------------------
@app.route("/journey/<image_name>")
def journey_map(image_name):
    postcard = None

    for item in data:
        if item.get("name") == image_name:
            postcard = item
            break

    if postcard is None:
        return "Postcard not found"

    origin_key = f"{postcard.get('origin_city','')}, {postcard.get('origin_country','')}"
    dest_key = f"{postcard.get('receiving_city','')}, {postcard.get('receiving_country','')}"

    if origin_key not in location_coords or dest_key not in location_coords:
        return "Coordinates not found for this postcard"

    start = location_coords[origin_key]
    end = location_coords[dest_key]

    m = folium.Map(tiles="CartoDB positron")

    folium.Marker(
        start,
        popup=f"From: {origin_key}",
        icon=folium.Icon(color="green")
    ).add_to(m)

    folium.Marker(
        end,
        popup=f"To: {dest_key}",
        icon=folium.Icon(color="red")
    ).add_to(m)

    curved_coords = curved_line(start, end, curve=0.35, points=120)

    AntPath(
        locations=curved_coords,
        color="#0047FF",
        pulse_color="orange",
        weight=7,
        delay=300,
        dash_array=[20, 30],
        opacity=0.9
    ).add_to(m)

    invisible = folium.PolyLine(
        curved_coords,
        opacity=0
    ).add_to(m)

    PolyLineTextPath(
        invisible,
        "➜",
        repeat=True,
        offset=8,
        attributes={
            "fill": "blue",
            "font-weight": "bold",
            "font-size": "20"
        }
    ).add_to(m)

    info = f"""
    <div style="
        position:fixed;
        top:20px;
        left:50%;
        transform: translateX(-50%);
        z-index:9999;
        background:white;
        padding:12px 20px;
        border-radius:8px;
        box-shadow:0 2px 8px rgba(0,0,0,.3);
        font-family:Arial;
        <b>Animated Postcard Journey</b><br><br>
        <b>Distance:</b> {postcard.get("distance")} km<br>
        <b>Travel Time:</b> {postcard.get("time")} days<br>
        <b>Sent:</b> {postcard.get("date_sent")}<br>
        <b>Received:</b> {postcard.get("date_received")}
    </div>
    """

    m.get_root().html.add_child(folium.Element(info))

    m.fit_bounds([start, end])
    return m._repr_html_()

# ----------------CLUSTERS-------------
@app.route("/clusters")
def clusters():
    selected_cluster = request.args.get("cluster", type=int)
    page = request.args.get("page",1, type=int)
    results = []
    if selected_cluster is not None:
        image_list = cluster_map.get(selected_cluster,[])
        for item in data:
            if item.get("name") in image_list:

                travel_time = item.get("time", 0)

                try:
                    travel_time = float(travel_time)
                except:
                    travel_time = 0


                results.append({
                    "name": item.get("name",""),
                    "origin":item.get("origin_country",""),
        
                    "destination":item.get("receiving_country",""),
                    "origin_city":item.get("origin_city",""),
                    "receiving_city":item.get("receiving_city",""),
                    "time": travel_time,
                    "date_sent":item.get("date_sent",""),
                    "date_received":item.get("date_received",""),
                    "distance":item.get("distance",0),
                    "outlier": travel_time >= 100
                })
    total_results = len(results)
    PER_PAGE = 42

    start = (page - 1) * PER_PAGE
    end = start + PER_PAGE

    paginated_results = results[start:end]

    has_more = end < total_results
    return render_template(
        "clusters.html",
        results=paginated_results,
        selected_cluster=selected_cluster,
        cluster_names=cluster_names,
        page=page,
        total_results=total_results,
        has_more=has_more
    )

@app.route("/topic-evolution")
def topic_evolution():
    abstraction = request.args.get("abstraction", "group")
    country1 = request.args.get("country1", "")
    country2 = request.args.get("country2", "")

    cluster_to_group = {}
    for group, topics in topic_hierarchy.items():
        for topic in topics:
            cluster_to_group[topic] = group

    def get_year(date_text):
        try:
            return datetime.strptime(date_text, "%Y-%m-%d").year
        except:
            return None

    def get_topic_name(item):
        name = item.get("name", "")
        cluster_id = image_to_cluster.get(name)

        if cluster_id is None:
            return "Unknown"

        cluster_topic = cluster_names.get(cluster_id, "Unknown")

        if abstraction == "group":
            return cluster_to_group.get(cluster_topic, "Other")
        return cluster_topic

    def country_matches(item, country):
        return (
            item.get("origin_country", "") == country or
            item.get("receiving_country", "") == country
        )

    all_countries = sorted(set(
        [i.get("origin_country", "") for i in data] +
        [i.get("receiving_country", "") for i in data]
    ))

    selected_countries = []
    if country1:
        selected_countries.append(country1)
    if country2:
        selected_countries.append(country2)

    if not selected_countries:
        selected_countries = ["All Countries"]

    country_topic_year_count = {}

    for country in selected_countries:
        country_topic_year_count[country] = {}

        for item in data:
            if country != "All Countries" and not country_matches(item, country):
                continue

            year = get_year(item.get("date_sent", ""))
            if year is None:
                continue

            topic = get_topic_name(item)

            if topic not in country_topic_year_count[country]:
                country_topic_year_count[country][topic] = {}

            country_topic_year_count[country][topic][year] = (
                country_topic_year_count[country][topic].get(year, 0) + 1
            )

    all_years = sorted(set(
        year
        for country_data in country_topic_year_count.values()
        for topic_data in country_data.values()
        for year in topic_data.keys()
    ))

    if len(selected_countries) == 1:
        fig = go.Figure()
        country = selected_countries[0]

        for topic, year_counts in country_topic_year_count[country].items():
            y_values = [year_counts.get(year, 0) for year in all_years]

            fig.add_trace(go.Scatter(
                x=all_years,
                y=y_values,
                mode="lines",
                stackgroup="one",
                name=topic,
                hovertemplate=(
                    f"<b>Country:</b> {country}<br>"
                    "<b>Topic:</b> %{fullData.name}<br>"
                    "<b>Year:</b> %{x}<br>"
                    "<b>Postcards:</b> %{y}<extra></extra>"
                )
            ))

        title = f"Topic Evolution Stream Graph - {country}"

    else:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            subplot_titles=(selected_countries[0], selected_countries[1])
        )

        for row, country in enumerate(selected_countries, start=1):
            for topic, year_counts in country_topic_year_count[country].items():
                y_values = [year_counts.get(year, 0) for year in all_years]

                fig.add_trace(go.Scatter(
                    x=all_years,
                    y=y_values,
                    mode="lines",
                    stackgroup=f"group{row}",
                    name=topic,
                    hovertemplate=(
                        f"<b>Country:</b> {country}<br>"
                        "<b>Topic:</b> %{fullData.name}<br>"
                        "<b>Year:</b> %{x}<br>"
                        "<b>Postcards:</b> %{y}<extra></extra>"
                    ),
                    showlegend=True if row == 1 else False
                ), row=row, col=1)

        title = f"Topic Evolution Comparison: {selected_countries[0]} vs {selected_countries[1]}"

    fig.update_layout(
        title=dict(
            text=(
                title +
                f"<br><sup>Abstraction level: {abstraction.title()} | "
                "Stream graph showing postcard topic changes over time</sup>"
            ),
            x=0.5
        ),
        height=850,
        hovermode="x unified",
        margin=dict(l=40, r=40, t=100, b=40),
        xaxis_title="Year",
        yaxis_title="Number of postcards"
    )

    controls = f"""
    <div style="
        padding:15px;
        font-family:Arial;
        background:#f4f8ff;
        border-bottom:1px solid #d0d7e2;
    ">
        <a href="/" style="
            background:#0077cc;
            color:white;
            padding:7px 14px;
            text-decoration:none;
            border-radius:5px;
            font-weight:bold;
            margin-right:12px;
        ">🏠 Home</a>

        <a href="/map" style="
            background:#0077cc;
            color:white;
            padding:7px 14px;
            text-decoration:none;
            border-radius:5px;
            font-weight:bold;
            margin-right:20px;
        ">🌍 Postcrossing Map</a>

        <form method="GET" style="display:inline-block;">
            <label><b>Topic abstraction:</b></label>
            <select name="abstraction" style="padding:6px;">
                <option value="group" {"selected" if abstraction == "group" else ""}>
                    High-level topic groups
                </option>
                <option value="cluster" {"selected" if abstraction == "cluster" else ""}>
                    Detailed clusters
                </option>
            </select>

            <label style="margin-left:20px;"><b>Country 1:</b></label>
            <select name="country1" style="padding:6px;">
                <option value="">All Countries</option>
    """

    for c in all_countries:
        controls += f'<option value="{c}" {"selected" if c == country1 else ""}>{c}</option>'

    controls += """
            </select>

            <label style="margin-left:20px;"><b>Country 2:</b></label>
            <select name="country2" style="padding:6px;">
                <option value="">None</option>
    """

    for c in all_countries:
        controls += f'<option value="{c}" {"selected" if c == country2 else ""}>{c}</option>'

    controls += """
            </select>

            <button type="submit" style="
                margin-left:20px;
                padding:7px 14px;
                background:#0077cc;
                color:white;
                border:none;
                border-radius:5px;
                font-weight:bold;
                cursor:pointer;
            ">Update</button>
        </form>

        <p style="margin:12px 0 0 0;color:#444;font-size:14px;text-align:center;">
            This stream graph shows how postcard topics change over time.
            Switch between high-level topic groups and detailed clusters,
            and compare two countries side by side.
        </p>
    </div>
    """

    return controls + fig.to_html(full_html=False, include_plotlyjs=True)


if __name__ =="__main__":
    app.run(debug=True)
