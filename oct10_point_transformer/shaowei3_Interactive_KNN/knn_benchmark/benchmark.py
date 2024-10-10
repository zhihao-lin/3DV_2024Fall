import os
import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from sklearn.neighbors import BallTree
from annoy import AnnoyIndex
from datasketch import MinHashLSH, MinHash
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt



def brute_force_knn(data, query_point, k=5):
    start = time.time()
    distances = cdist([query_point], data)
    nearest_indices = np.argsort(distances[0])[:k]
    duration = time.time() - start
    return nearest_indices, distances[0][nearest_indices], duration


def kd_tree_knn(data, query_point, k=5):
    tree = KDTree(data)
    start = time.time()
    distances, indices = tree.query(query_point, k=k)
    duration = time.time() - start
    return indices, distances, duration


def ann_knn(data, query_point, k=5):
    dim = len(query_point)
    ann = AnnoyIndex(dim, 'euclidean')
    for i, point in enumerate(data):
        ann.add_item(i, point)

    ann.build(10)

    start = time.time()
    nearest = ann.get_nns_by_vector(query_point, k, include_distances=True)
    duration = time.time() - start

    return nearest[0], nearest[1], duration


def ball_tree_knn(data, query_point, k=5):
    tree = BallTree(data)
    start = time.time()
    distances, indices = tree.query([query_point], k=k)
    duration = time.time() - start

    return indices.flatten(), distances.flatten(), duration


def hash_knn(data, query_point, k=5):
    lsh = MinHashLSH(num_perm=128, threshold=0.5)

    data = normalize(data)
    query_point = normalize([query_point])[0]

    minhashes = []
    for i, point in enumerate(data):
        m = MinHash(num_perm=128)
        for coord in point:
            m.update(coord.tobytes())
        lsh.insert(i, m)
        minhashes.append(m)

    query_m = MinHash(num_perm=128)
    for coord in query_point:
        query_m.update(coord.tobytes())

    start = time.time()
    result = lsh.query(query_m)
    nearest = result[:k]
    distances = np.linalg.norm(data[nearest] - query_point, axis=1)
    duration = time.time() - start

    return nearest, distances, duration



def benchmark(method, data, query_point, k=5):
    indices, distances, duration = method(data, query_point, k)
    return indices, distances, duration



def plot_benchmark_results(methods, times, output_file):
    plt.figure(figsize=(10, 6))
    plt.bar(methods, times, color='skyblue')
    plt.xlabel('kNN Method')
    plt.ylabel('Running Time (seconds)')
    plt.title('Benchmark of Different kNN Methods')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Figure saved as {output_file}")
    plt.close()  # Close the plot after saving


def run_benchmarks(data, query_point, k=5, output_file='benchmark.png'):
    methods = {
        "Brute Force": brute_force_knn,
        "KD-Tree": kd_tree_knn,
        "Ball-Tree": ball_tree_knn,
        "ANN": ann_knn,
        "Hashing (LSH)": hash_knn
    }

    times = []
    method_names = []

    results = {}
    for name, method in methods.items():
        indices, distances, duration = benchmark(method, data, query_point, k)
        times.append(duration)
        method_names.append(name)
        results[name] = {
            'time': duration,
            'indices': indices,
            'distances': distances
        }
        
    plot_benchmark_results(method_names, times, output_file)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_size', type=int, default=10000, help='number of data points')
    parser.add_argument('--N', type=int, default=5, help='data dimensions')
    parser.add_argument('--save_dir', type=str, default='figs', help='directory to save the figure')
    args = parser.parse_args()
    N = args.N
    data_size = args.data_size
    data = np.random.rand(data_size, N)

    
    query_point = np.random.rand(N)
    k = 5

    # Run benchmarks
    os.makedirs(args.save_dir, exist_ok=True)
    output_file = os.path.join(args.save_dir, 'benchmark_{}pts_{}dim.png'.format(data_size, N))
    results = run_benchmarks(data, query_point, k, output_file)
