import streamlit as st
import folium
import numpy as np
import pandas as pd
import itertools
from streamlit_folium import folium_static
from scipy.spatial.distance import pdist, squareform
from geopy.distance import geodesic
import random
import math
import time

# Function to calculate the distance between two points using geopy
def calculate_distance(point1, point2):
    return geodesic(point1, point2).kilometers

# Function to generate a distance matrix for a list of points
def generate_distance_matrix(points):
    distance_matrix = np.zeros((len(points), len(points)))
    for i, point1 in enumerate(points):
        for j, point2 in enumerate(points):
            distance_matrix[i][j] = calculate_distance(point1, point2)
    return distance_matrix

# Function to visualize a distance matrix as a heatmap
def visualize_distance_matrix(distance_matrix):
    st.write("Distance Matrix:")
    st.write(pd.DataFrame(distance_matrix, columns=range(1, len(distance_matrix)+1), index=range(1, len(distance_matrix)+1)))

# Function to solve the TSP using the nearest neighbor algorithm
def nearest_neighbor_algorithm(points):
    if len(points) < 2:
        return points
    start_point = points[0]
    solution = [start_point]
    current_point = start_point
    remaining_points = set(points[1:])

    while remaining_points:
        nearest_point = min(remaining_points, key=lambda point: np.linalg.norm(np.array(point) - np.array(current_point)))
        remaining_points.remove(nearest_point)
        solution.append(nearest_point)
        current_point = nearest_point

    return solution + [start_point]  # return to start and close the loop

# Function to solve the TSP using the brute force algorithm
def brute_force_algorithm(points):
    shortest_path = None
    min_distance = float('inf')
    for perm in itertools.permutations(points):
        total_distance = sum(calculate_distance(perm[i], perm[i+1]) for i in range(len(perm)-1))
        total_distance += calculate_distance(perm[-1], perm[0])  # add distance from last point back to the start
        if total_distance < min_distance:
            min_distance = total_distance
            shortest_path = perm + (perm[0],)  # add the first point to close the loop
    return shortest_path

# Function to solve the TSP using random sampling algorithm
def random_sampling_algorithm(points):
    shortest_path = None
    min_distance = float('inf')
    num_samples = min(1000, len(points)**2)  # Limit the number of samples to avoid excessive computation
    for _ in range(num_samples):
        sample_path = random.sample(points, len(points))
        total_distance = sum(calculate_distance(sample_path[i], sample_path[i+1]) for i in range(len(sample_path)-1))
        total_distance += calculate_distance(sample_path[-1], sample_path[0])  # add distance from last point back to the start
        if total_distance < min_distance:
            min_distance = total_distance
            shortest_path = sample_path + [sample_path[0]]  # add the first point to close the loop
    return shortest_path

# Function to solve the TSP using genetic algorithm
def genetic_algorithm(points, population_size=50, generations=100):
    def create_individual(points):
        return random.sample(points, len(points))

    def fitness(individual):
        return sum(calculate_distance(individual[i], individual[i+1]) for i in range(len(individual)-1))

    def crossover(parent1, parent2):
        crossover_point = random.randint(0, len(parent1)-1)
        child = parent1[:crossover_point] + [gene for gene in parent2 if gene not in parent1[:crossover_point]]
        return child

    def mutate(individual):
        idx1, idx2 = random.sample(range(len(individual)), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual

    population = [create_individual(points) for _ in range(population_size)]
    for _ in range(generations):
        # Select parents
        parents = random.choices(population, k=2, weights=[fitness(individual) for individual in population])

        # Crossover
        child = crossover(parents[0], parents[1])

        # Mutate
        if random.random() < 0.1:  # Mutation probability
            child = mutate(child)

        # Replace worst individual in the population
        worst_idx = max(range(population_size), key=lambda i: fitness(population[i]))
        population[worst_idx] = child

    return min(population, key=fitness) + [min(population, key=fitness)[0]]  # add the first point to close the loop

# Function to solve the TSP using the simulated annealing algorithm
def simulated_annealing_algorithm(points):
    current_solution = nearest_neighbor_algorithm(points)  # Start with a solution from Nearest Neighbor
    current_solution = current_solution[:-1]  # Remove the duplicated starting point
    initial_temperature = 1000
    cooling_rate = 0.003

    current_cost = sum(calculate_distance(current_solution[i], current_solution[i+1]) for i in range(len(current_solution)-1))
    current_cost += calculate_distance(current_solution[-1], current_solution[0])  # add distance from last point back to the start

    temperature = initial_temperature
    while temperature > 1:
        new_solution = current_solution[:]
        idx1, idx2 = random.sample(range(len(new_solution)), 2)
        new_solution[idx1], new_solution[idx2] = new_solution[idx2], new_solution[idx1]
        new_cost = sum(calculate_distance(new_solution[i], new_solution[i+1]) for i in range(len(new_solution)-1))
        new_cost += calculate_distance(new_solution[-1], new_solution[0])  # add distance from last point back to the start

        if new_cost < current_cost or random.random() < math.exp((current_cost - new_cost) / temperature):
            current_solution = new_solution
            current_cost = new_cost

        temperature *= 1 - cooling_rate

    return current_solution + [current_solution[0]]  # add the first point to close the loop

# Function to add markers and draw path on the map
def add_markers_and_path(map_object, points, path):
    for point in points:
        folium.Marker(
            location=[point[0], point[1]],
            popup=f'({point[0]}, {point[1]})'
        ).add_to(map_object)
    # Draw the path if available
    if path:
        folium.PolyLine(path, color='blue', weight=5, opacity=0.7).add_to(map_object)


# Streamlit app
def app():
    st.title('Traveling Salesperson Problem Solver')

    # Initialize or update the session state for points
    if 'points' not in st.session_state:
        st.session_state.points = []

    # Form for adding new markers
    with st.form("points_input_add"):
        lat = st.number_input('Latitude', value=33.8704, format="%.4f")
        lon = st.number_input('Longitude', value=-117.9242, format="%.4f")
        submitted = st.form_submit_button('Add Marker')
        if submitted:
            st.session_state.points.append((lat, lon))

    # Display map with current markers
    m = folium.Map(location=[33.8704, -117.9242], zoom_start=13, tiles='OpenStreetMap')
    add_markers_and_path(m, st.session_state.points, [])
    folium_static(m)

    # Checkboxes for TSP algorithms
    tsp_algorithms = {
        "Nearest Neighbor": nearest_neighbor_algorithm,
        "Brute Force": brute_force_algorithm,
        "Random Sampling": random_sampling_algorithm,
        "Genetic Algorithm": genetic_algorithm,
        "Simulated Annealing": simulated_annealing_algorithm
    }
    selected_algorithms = st.multiselect("Select TSP Algorithms", list(tsp_algorithms.keys()))

    # Button to compute the optimized route
    if st.button('Compute Optimized Route'):
        if len(st.session_state.points) >= 2 and selected_algorithms:
            execution_times = {}
            routes = {}
            best_route = None
            best_algorithm = None
            min_distance = float('inf')
            for algorithm in selected_algorithms:
                start_time = time.time()
                route = tsp_algorithms[algorithm](st.session_state.points)
                end_time = time.time()
                execution_time = end_time - start_time
                execution_times[algorithm] = execution_time
                routes[algorithm] = route

                total_distance = sum(calculate_distance(route[i], route[i+1]) for i in range(len(route)-1))
                total_distance += calculate_distance(route[-1], route[0])  # add distance from last point back to the start
                
                if total_distance < min_distance:
                    min_distance = total_distance
                    best_route = route
                    best_algorithm = algorithm

                st.write(f"Optimized Delivery Route ({algorithm}):")
                st.write(f"Total Distance: {total_distance} kilometers")
                if algorithm == "Brute Force":
                    # Display the optimized route as a closed loop
                    for idx, loc in enumerate(route):
                        st.write(f"{idx+1}: {loc}")
                else:
                    for idx, loc in enumerate(route):
                        st.write(f"{idx+1}: {loc}")

                # Generate distance matrix
                distance_matrix = generate_distance_matrix(route)
                visualize_distance_matrix(distance_matrix)

                # Re-draw the map with the route
                m = folium.Map(location=[33.8704, -117.9242], zoom_start=13, tiles='OpenStreetMap')
                add_markers_and_path(m, st.session_state.points, route)
                folium_static(m)
            
            # Print the best route and its total distance
            st.write(f"Best Optimized Delivery Route ({best_algorithm}):")
            for idx, loc in enumerate(best_route):
                st.write(f"{idx+1}: {loc}")
            total_distance_best = sum(calculate_distance(best_route[i], best_route[i+1]) for i in range(len(best_route)-1))
            total_distance_best += calculate_distance(best_route[-1], best_route[0])  # add distance from last point back to the start
            st.write(f"Total Distance: {total_distance_best} kilometers")

            # Re-draw the map with the best route
            m = folium.Map(location=[33.8704, -117.9242], zoom_start=13, tiles='OpenStreetMap')
            add_markers_and_path(m, st.session_state.points, best_route)
            folium_static(m)
            
            # Bar graph for execution times
            st.subheader("Execution Times of TSP Algorithms")
            st.bar_chart(pd.DataFrame(execution_times.items(), columns=['Algorithm', 'Execution Time (s)']))

        else:
            st.error("Please add at least two locations and select at least one algorithm.")

    # Refresh map and clear session state
    if st.button('Refresh Map'):
        st.session_state.points = []
        m = folium.Map(location=[33.8704, -117.9242], zoom_start=13, tiles='OpenStreetMap')
        folium_static(m)

if __name__ == "__main__":
    app()