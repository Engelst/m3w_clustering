import argparse
import numpy as np
from BorderPeel import BorderPeel
from device_fix import ensure_tensor, DEVICE
import clustering_tools as ct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--n-clusters', type=int, required=True)
    parser.add_argument('--no-labels', action='store_true')
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")

    # 加载数据
    print("Loading data...")
    data, labels = ct.read_data(args.input, has_labels=not args.no_labels)
    print(f"Loaded data shape: {data.shape}")

    if not args.no_labels:
        print(f"Number of true clusters: {len(np.unique(labels))}")

    # 数据标准化
    print("Standardizing data...")
    data = ensure_tensor(data)

    # M3W配置
    m3w_config = {
        'k': 7,
        'max_iterations': 10,
        'mean_border_eps': 0.01,
        'verbose': True,
        'n_clusters': args.n_clusters  # 传递聚类数量参数
    }

    print("Running M3W clustering...")
    m3w = BorderPeel(**m3w_config)
    border_values = m3w.fit(data)

    # 保存结果
    print("Saving results...")
    np.savetxt(args.output, border_values, fmt='%d')

    print("Done!")


if __name__ == "__main__":
    main()