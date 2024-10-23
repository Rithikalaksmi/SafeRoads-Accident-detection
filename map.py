import warnings
import pandas as pd
import folium
from sklearn.cluster import KMeans
from shapely.geometry import MultiPoint

# Suppress all warnings
warnings.filterwarnings('ignore')

# Load the dataset
data_kmeans = pd.read_csv(r'C:\Users\rithi\Downloads\accidents_2012_to_2014.csv (1)\accidents_2012_to_2014.csv', low_memory=False)

# Preprocessing: Drop rows with NaN values in Longitude, Latitude, or Severity
data_kmeans = data_kmeans.dropna(subset=['Longitude', 'Latitude', 'Accident_Severity'])

# Sample 10,000 rows for efficiency
data_kmeans = data_kmeans.sample(n=10000, random_state=42)

# Step 1: Run K-Means with k = 5
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
data_kmeans['Zone'] = kmeans.fit_predict(data_kmeans[['Longitude', 'Latitude']])

# Step 2: Visualize the clustered map
# Create a folium map centered in the UK
uk_map = folium.Map(location=[54.5, -4.0], zoom_start=6)  # Center of the UK

# Define colors for each zone
zone_colors = {
    0: 'blue',    # Zone 0
    1: 'orange',  # Zone 1
    2: 'purple',  # Zone 2
    3: 'green',   # Zone 3
    4: 'red',     # Zone 4
}

# Define colors for severity
severity_colors = {
    1: 'red',    # Fatal
    2: 'yellow', # Serious
    3: 'green',  # Slight
    4: 'blue',   # Other (customize as needed)
}

# Add borders for each zone based on zone colors
for zone in range(k):
    zone_data = data_kmeans[data_kmeans['Zone'] == zone]

    # Create a polygon around the points in the zone
    if len(zone_data) > 2:  # Need at least 3 points to form a polygon
        hull = MultiPoint(zone_data[['Longitude', 'Latitude']].values).convex_hull
        hull_coords = [(point[1], point[0]) for point in hull.exterior.coords]

        # Add a polygon for the zone border with a darker color
        folium.Polygon(
            locations=hull_coords,
            color=zone_colors[zone],
            weight=3,  # Darker and thicker border
            fill=False,  # Fill set to False to show only the border
            opacity=0.9,  # Increased opacity for visibility
        ).add_to(uk_map)

# Add circles to the map based on clusters and severity
for _, row in data_kmeans.iterrows():
    folium.CircleMarker(
        location=(row['Latitude'], row['Longitude']),
        radius=5,
        color=severity_colors.get(row['Accident_Severity'], 'gray'),  # Default to gray if severity not found
        fill=True,
        fill_color=severity_colors.get(row['Accident_Severity'], 'gray'),
        fill_opacity=0.6,
        popup=f"Zone: {row['Zone']}<br>Severity: {row['Accident_Severity']}",
    ).add_to(uk_map)

# Save the map to the static folder for access in the HTML page
uk_map.save('C:/Users/rithi/OneDrive/Documents/sem 5/machine learning lab/project/Front end final/Front end/static/maps/accident_zones_map_with_markers.html')

print("Map has been saved successfully!")
