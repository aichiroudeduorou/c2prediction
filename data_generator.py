import numpy as np
import networkx as nx
import os
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
import warnings

warnings.filterwarnings("ignore")

np.random.seed(42)




# edge mapping (4 types)
def get_edge_mechanism():
    """Define a random edge function type and parameters"""
    func_type = np.random.choice(['nn', 'discretize', 'tree', 'noise'], p=[0.4, 0.2, 0.2, 0.2])
    params = {'type': func_type}

    if func_type == 'nn':
        # Small neural network (with random activation)
        params['W'] = np.random.randn(1, 1) * 0.5 + 0.5
        params['b'] = np.random.randn(1) * 0.1
        params['act'] = np.random.choice(['relu', 'tanh', 'sin'])

    elif func_type == 'discretize':
        params['n_bins'] = np.random.randint(2, 6)
        params['embedding'] = np.random.randn(1)
        params['scale'] = np.random.uniform(0.5, 2.0)

    elif func_type == 'tree':
        # Decision tree rule: single split
        params['threshold'] = np.random.uniform(-1, 1)
        params['direction'] = np.random.choice([1, -1])

    # 'noise' needs no params
    return params


def apply_edge_mechanism(x, params):
    """Apply the fixed edge function to input x using params"""
    func_type = params['type']
    val = x[0]

    if func_type == 'nn':
        W = params['W']
        b = params['b']
        z = W @ x + b
        act = params['act']
        if act == 'relu':
            return np.maximum(0, z)
        elif act == 'tanh':
            return np.tanh(z)
        else:  # sin
            return np.sin(z)

    elif func_type == 'discretize':
        # Discretization
        n_bins = params['n_bins']
        val_disc = np.round(val * params['scale'] * n_bins)
        return params['embedding'] * (val_disc / n_bins)

    elif func_type == 'tree':
        # Decision tree rule
        threshold = params['threshold']
        noise = np.random.randn(len(x)) * 0.05
        if val > threshold:
            return np.array([val + 0.5 * params['direction']]) + noise
        else:
            return np.array([-val + 0.5 * params['direction']]) + noise

    else:  # 'noise'
        return x + np.random.randn(len(x)) * 0.05



# Generate a synthetic dataset based on SCM
def generate_scm_dataset(
        n_samples=100,
        n_features=10,
        max_nodes=20,
        gnr_p=0.3,
        task='regression'  # or 'classification'
):
    """
    Generate a synthetic dataset based on SCM (Structural Causal Model)
    Returns: X (n_samples, n_features), y (n_samples,)
    """
    # 1. Generate a causal graph (DAG)
    # Ensure we have enough nodes for features + target
    if max_nodes <= n_features:
        max_nodes = n_features + 2 # Add a buffer
    
    # Generate graph with size at least n_features + 1 (for target)
    N = np.random.randint(n_features + 1, max_nodes + 1)
    G = nx.gnr_graph(N, p=gnr_p, seed=None)
    
    # Define features and target BEFORE the loop to ensure consistency across samples
    nodes = list(G.nodes)
    
    # Select target node (usually a leaf or a sink node is harder, random is fine)
    target_node = np.random.choice(nodes)
    
    # Select feature nodes (exclude target to simulate X -> y or y -> X prediction without leakage)
    # If you want the target to be part of the graph structure but not an input feature:
    available_nodes = [n for n in nodes if n != target_node]
    
    # If we don't have enough nodes left (unlikely given N >= n_features + 1)
    if len(available_nodes) < n_features:
        # Fallback: allow overlap or just take what we have
        feature_nodes = np.array(available_nodes)
    else:
        feature_nodes = np.random.choice(available_nodes, size=n_features, replace=False)
        
    # Sort feature nodes to keep column order consistent
    feature_nodes = sorted(list(feature_nodes))

    # Assign mechanisms to edges
    for u, v in G.edges():
        G[u][v]['mechanism'] = get_edge_mechanism()

    # 2. Generate each sample in topological order
    all_X = []
    all_y = []

    for _ in range(n_samples):
        node_values = {}

        # Initialize root nodes
        for node in nx.topological_sort(G):
            if G.in_degree(node) == 0:
                    # Root node: scalar noise (simplified to 1D)
                noise_type = np.random.choice(['normal', 'uniform'])
                if noise_type == 'normal':
                    node_values[node] = np.random.randn()
                else:
                    node_values[node] = np.random.uniform(-1, 1)
            else:
                # Aggregate parent nodes (sum)
                parents = list(G.predecessors(node))
                parent_sum = sum(node_values[p] for p in parents)
                # Apply edge function (simplified: apply independently to each parent, then sum)
                transformed = 0.0
                for p in parents:
                    x = np.array([node_values[p]])  # Temporarily expand to vector
                    # For simplicity, only handle scalars here, but simulate vector behavior
                    params = G[p][node]['mechanism']
                    val = apply_edge_mechanism(x, params)[0]
                    transformed += val
                node_values[node] = transformed + np.random.randn() * 0.01

        # 3. Extract features and target
        # (consistent selection across samples)
        X_sample = np.array([node_values.get(n, 0.0) for n in feature_nodes])
        y_val = node_values.get(target_node, 0.0)

        all_X.append(X_sample)
        all_y.append(y_val)

    X = np.array(all_X)  # (n_samples, n_features)
    y = np.array(all_y)  # (n_samples,)

    # 4. Post-processing: Kumaraswamy warping (only for non-target features)
    def kumaraswamy_warp(x, a=2.0, b=2.0):
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)  # Normalize to [0,1]
        x = np.clip(x, 1e-6, 1 - 1e-6)
        return 1 - (1 - x ** a) ** b

    X = kumaraswamy_warp(X)

    # 5. Classification task: discretize y
    if task == 'classification':
        # n_classes = np.random.randint(2, 6)
        n_classes = 2  # For binary classification
        disc_y = KBinsDiscretizer(n_bins=n_classes, encode='ordinal', strategy='quantile')
        y = disc_y.fit_transform(y.reshape(-1, 1)).flatten().astype(int)

    # 6. Random missing values (MCAR)
    # missing_rate = 0.05
    # mask = np.random.rand(*X.shape) < missing_rate
    # X[mask] = np.nan

    # Visualize graph structure

    # import matplotlib
    # matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    save_dir = f"/workspace/causal_discovery/dataset/syn_dataset/{n_samples}samples_{n_features}features"
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    nx.draw(G, ax=ax, with_labels=True, node_color='lightblue', node_size=300, font_size=10)
    ax.set_title("Sample SCM DAG (GNR)")
    img_path = os.path.join(save_dir, "scm_dag.png")
    fig.savefig(img_path, dpi=150)
    print(f"\n✅ Causal graph saved as '{img_path}'")

    # Save adjacency matrix of the causal graph as a labeled 0-1 CSV file
    adj = nx.to_numpy_array(G, dtype=int)
    node_labels = [str(n) for n in G.nodes]
    adj_df = pd.DataFrame(adj, columns=node_labels, index=node_labels)
    adj_df.index.name = 'labels'
    adj_csv_path = os.path.join(save_dir, "ground_truth.csv")
    adj_df.to_csv(adj_csv_path)
    print(f"✅ Causal graph adjacency matrix saved as '{adj_csv_path}'")

    return X, y



# ==============================
# Test generation
# ==============================

if __name__ == "__main__":
    # Ensure output directory exists
    n_samples=10000
    n_features=400
    save_dir = f"/workspace/causal_discovery/dataset/syn_dataset/{n_samples}samples_{n_features}features"
    os.makedirs(save_dir, exist_ok=True)

    # print("🔄 Generating regression task data...")
    # X_reg, y_reg = generate_scm_dataset(n_samples=n_samples, n_features=n_features, task='regression')
    # print(f"Regression data shape: X={X_reg.shape}, y={y_reg.shape}")
    # print(f"First 5 y values: {y_reg[:5]}")

    print("\n🔄 Generating classification task data...")
    X_clf, y_clf = generate_scm_dataset(n_samples=n_samples, n_features=n_features, task='classification')
    print(f"Classification data shape: X={X_clf.shape}, y={y_clf.shape}")
    print(f"Class distribution: {np.bincount(y_clf)}")

    # Save classification data to CSV
    print("\n💾 Saving classification data...")
    feature_names = [f"feature_{i}" for i in range(X_clf.shape[1])]
    df_clf = pd.DataFrame(X_clf, columns=feature_names)
    df_clf['target'] = y_clf

    csv_path = os.path.join(save_dir, "scm_classification.csv")
    df_clf.to_csv(csv_path, index=False)
    print(f"✅ CSV saved to: {csv_path}")

    