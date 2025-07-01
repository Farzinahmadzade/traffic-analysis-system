import logging
from itertools import islice
from typing import Dict, List, Tuple

import folium
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from folium.plugins import HeatMap
from sklearn.cluster import KMeans

# --- Constants ---
# Using constants makes the code easier to read and modify.
TEHRAN_COORDS = (35.6892, 51.3890)
NUM_CLUSTERS = 3
K_ALTERNATIVE_ROUTES = 3  # Find the top 3 shortest paths
TRAFFIC_WEIGHT = 0.4
DEMAND_WEIGHT = 0.6
MIN_ZONE_DISTANCE = 0.01  # Minimum distance between zone points to avoid clutter
RANDOM_STATE = 42
ZONE_COLORS = {"red": "red", "green": "green", "yellow": "orange"}


# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def k_shortest_paths(
    G: nx.Graph, source: int, target: int, k: int, weight: str = "length"
) -> List[List[int]]:
    """
    Returns the k-shortest simple paths from source to target in a graph G.
    This is a wrapper around networkx's `shortest_simple_paths`.
    """
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))


class TrafficManager:
    """
    Manages fetching network data, simulating traffic, and identifying traffic hot zones.
    """

    def __init__(self, place: str = "Tehran, Iran"):
        self.place = place
        self.graph: nx.Graph | None = None
        self.nodes: pd.DataFrame | None = None
        self.edges: pd.DataFrame | None = None
        self.traffic_data: pd.DataFrame | None = None
        self.demand_data: pd.DataFrame | None = None
        self.zone_data: pd.DataFrame | None = None
        self.heatmap_data: pd.DataFrame | None = None

    def initialize_network(self):
        """Fetches and initializes the street network graph for the specified place."""
        try:
            logger.info(f"Fetching network data for {self.place}...")
            self.graph = ox.graph_from_place(self.place, network_type="drive")
            self.nodes, self.edges = ox.graph_to_gdfs(self.graph)
            logger.info("Network data initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize network: {e}")
            raise

    def _generate_simulation_data(self):
        """Generates simulated traffic and demand data for the network."""
        if self.nodes is None:
            raise ValueError("Network must be initialized before generating data.")

        node_count = len(self.nodes)
        rng = np.random.default_rng(RANDOM_STATE)
        traffic_values = rng.uniform(0, 100, node_count)
        demand_values = rng.uniform(0, 100, node_count)

        self.heatmap_data = pd.DataFrame(
            {"lat": self.nodes["y"], "lon": self.nodes["x"], "traffic": traffic_values}
        )
        self.traffic_data = self.heatmap_data
        self.demand_data = pd.DataFrame({"lat": self.nodes["y"], "lon": self.nodes["x"], "demand": demand_values})

    def identify_hot_zones(self):
        """Identifies high-traffic and high-demand zones using KMeans clustering."""
        self._generate_simulation_data()
        if self.traffic_data is None or self.demand_data is None:
            raise ValueError("Traffic and demand data must be available.")

        combined_data = pd.merge(self.traffic_data, self.demand_data, on=["lat", "lon"])
        combined_data["zone_score"] = (
            TRAFFIC_WEIGHT * combined_data["traffic"] + DEMAND_WEIGHT * combined_data["demand"]
        )

        kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=RANDOM_STATE, n_init="auto")
        combined_data["cluster"] = kmeans.fit_predict(combined_data[["zone_score"]])

        # --- Dynamically map clusters to zone types ---
        # This is more robust than assuming cluster 0 is always "high", 1 is "medium", etc.
        cluster_centers = kmeans.cluster_centers_.flatten()
        sorted_centers = np.argsort(cluster_centers)
        zone_type_map = {sorted_centers[0]: "green", sorted_centers[1]: "yellow", sorted_centers[2]: "red"}
        combined_data["zone_type"] = combined_data["cluster"].map(zone_type_map)

        self.zone_data = self._filter_proximal_points(combined_data, MIN_ZONE_DISTANCE)
        return self.zone_data

    @staticmethod
    def _filter_proximal_points(zoned_data: pd.DataFrame, min_dist: float) -> pd.DataFrame:
        """Filters out points that are too close to each other within the same zone for cleaner visualization."""
        filtered_indices = []
        for zone_type in zoned_data["zone_type"].unique():
            zone_points = zoned_data[zoned_data["zone_type"] == zone_type]
            coords = zone_points[["lat", "lon"]].to_numpy()
            keep_indices = []
            for i in range(len(coords)):
                is_far_enough = True
                for j in keep_indices:
                    # Using squared Euclidean distance is faster as it avoids the sqrt operation.
                    dist_sq = np.sum((coords[i] - coords[j]) ** 2)
                    if dist_sq < min_dist**2:
                        is_far_enough = False
                        break
                if is_far_enough:
                    keep_indices.append(i)
            filtered_indices.extend(zone_points.index[keep_indices])
        return zoned_data.loc[filtered_indices]

    def find_alternative_routes(self, start_point: Tuple[float, float], end_point: Tuple[float, float]) -> list:
        """Finds K-shortest paths between two geographic points."""
        if self.graph is None:
            raise ValueError("Network must be initialized first.")

        start_node = ox.nearest_nodes(self.graph, start_point[1], start_point[0])
        end_node = ox.nearest_nodes(self.graph, end_point[1], end_point[0])

        try:
            paths = k_shortest_paths(self.graph, start_node, end_node, k=K_ALTERNATIVE_ROUTES)
            return paths
        except nx.NetworkXNoPath:
            logger.warning(f"No path found between {start_point} and {end_point}.")
            return []

    def get_zone_recommendations(self) -> Dict[str, List[Tuple[float, float]]]:
        """Groups zone coordinates by their type (red, green, yellow)."""
        if self.zone_data is None:
            raise ValueError("Zone data must be calculated first.")

        recommendations = {"red": [], "green": [], "yellow": []}
        for _, row in self.zone_data.iterrows():
            zone_type = row["zone_type"]
            if zone_type in recommendations:
                recommendations[zone_type].append((row["lat"], row["lon"]))
        return recommendations


def create_traffic_map(
    heatmap_data: pd.DataFrame, edges: pd.DataFrame, zone_data: pd.DataFrame | None = None
) -> folium.Map:
    """Creates an interactive Folium map with traffic heatmap and zone markers."""
    traffic_map = folium.Map(location=TEHRAN_COORDS, zoom_start=12, tiles="cartodbdark_matter")

    HeatMap(
        data=heatmap_data[["lat", "lon", "traffic"]].to_numpy(),
        radius=25,
        blur=20,
        max_zoom=1,
        min_opacity=0.4,
    ).add_to(traffic_map)

    if zone_data is not None:
        for _, row in zone_data.iterrows():
            folium.CircleMarker(
                location=(row["lat"], row["lon"]),
                radius=8,
                color=ZONE_COLORS.get(row["zone_type"], "blue"),
                fill=True,
                fill_opacity=0.7,
                tooltip=f"Zone: {row['zone_type'].capitalize()}",
                popup=(
                    f"<b>{row['zone_type'].capitalize()} Zone</b><br>"
                    f"Traffic: {row['traffic']:.1f}<br>"
                    f"Demand: {row['demand']:.1f}"
                ),
            ).add_to(traffic_map)

    folium.LayerControl().add_to(traffic_map)
    return traffic_map


def main():
    """Main execution function to run the traffic analysis."""
    try:
        traffic_manager = TrafficManager()
        traffic_manager.initialize_network()
        zone_data = traffic_manager.identify_hot_zones()
        recommendations = traffic_manager.get_zone_recommendations()

        logger.info(f"Identified {len(recommendations['red'])} red zone points.")
        logger.info(f"Identified {len(recommendations['green'])} green zone points.")
        logger.info(f"Identified {len(recommendations['yellow'])} yellow zone points.")

        if traffic_manager.heatmap_data is not None and traffic_manager.edges is not None:
            traffic_map = create_traffic_map(traffic_manager.heatmap_data, traffic_manager.edges, zone_data)
            map_filename = "tehran_traffic_analysis.html"
            traffic_map.save(map_filename)
            logger.info(f"Map saved successfully as '{map_filename}'.")

        # Example of finding alternative routes
        if recommendations["red"] and recommendations["green"]:
            start_point = recommendations["red"][0]
            end_point = recommendations["green"][0]
            alternative_routes = traffic_manager.find_alternative_routes(start_point, end_point)
            logger.info(f"Found {len(alternative_routes)} alternative routes between a red and a green zone.")

    except Exception as e:
        logger.critical(f"A critical error occurred in the main execution block: {e}")
        raise


if __name__ == "__main__":
    main()