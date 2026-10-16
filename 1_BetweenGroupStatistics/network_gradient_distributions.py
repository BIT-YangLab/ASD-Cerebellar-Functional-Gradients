import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

plt.rcParams['figure.dpi'] = 300

# ==================== Configuration ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'ABIDE1', 'cere-cere', '7net')
COLOR_FILE = os.path.join(BASE_DIR, 'Yeo_7Networks_Color_exp5.mat')
OUTPUT_DIR = os.path.join(BASE_DIR, 'results')
NETWORK_NAMES = ['VN', 'SMN', 'DAN', 'VAN', 'FPN', 'DMN']
GROUPS = ['ASD', 'controls']


def curve(data):
    """Calculate the kernel density estimate of the input data."""
    kde = gaussian_kde(data)
    x = np.linspace(1.5 * np.min(data), 1.5 * np.max(data), 500)
    return x, kde(x)


def calculate_network_means():
    """Calculate the mean value of each network for both groups."""
    means = {
        'ASD': [],
        'Controls': [],
        'network_names': NETWORK_NAMES
    }

    print("=== Network means ===")

    for group in GROUPS:
        print(f"\n--- {group} ---")

        for i, network in enumerate(NETWORK_NAMES, start=1):
            filename = f"{group}_Net{i}.txt"
            file_path = os.path.join(DATA_DIR, filename)

            try:
                data = np.loadtxt(file_path)
                mean_value = np.mean(data)
                means['ASD' if group == 'ASD' else 'Controls'].append(mean_value)
                print(f"{network}: mean = {mean_value:.4f}")
            except (OSError, ValueError) as e:
                print(f"Could not read {filename}: {e}")
                means['ASD' if group == 'ASD' else 'Controls'].append(np.nan)

    return means


def compare_asd_movement(means):
    """Compare network mean values between ASD and control groups."""
    asd_means = means['ASD']
    control_means = means['Controls']
    network_names = means['network_names']

    right_moves = 0
    left_moves = 0
    no_moves = 0

    print("\n=== Group differences ===")

    for i, network in enumerate(network_names):
        asd_mean = asd_means[i]
        control_mean = control_means[i]

        if np.isnan(asd_mean) or np.isnan(control_mean):
            print(f"{network}: insufficient data for comparison")
            continue

        difference = asd_mean - control_mean

        if difference > 0:
            direction = "higher in ASD"
            right_moves += 1
        elif difference < 0:
            direction = "lower in ASD"
            left_moves += 1
        else:
            direction = "no difference"
            no_moves += 1

        print(
            f"{network}: {direction} "
            f"(ASD = {asd_mean:.4f}, Controls = {control_mean:.4f}, "
            f"difference = {difference:.4f})"
        )

    print("\nDirection counts:")
    print(f"Higher in ASD: {right_moves}")
    print(f"Lower in ASD: {left_moves}")
    print(f"No difference: {no_moves}")

    return {
        'asd_means': asd_means,
        'control_means': control_means,
        'network_names': network_names,
        'right_moves': right_moves,
        'left_moves': left_moves,
        'no_moves': no_moves
    }


def main():
    fig, ax = plt.subplots(figsize=(4, 3.5))

    color = scipy.io.loadmat(COLOR_FILE)['color_exp5']

    # Vertical spacing between networks and groups
    spacing = 2
    gap = 1.8

    y_positions = (
        [i * spacing for i in range(6)] +
        [6 * spacing + gap + i * spacing for i in range(6)]
    )

    # Plot networks in reverse order within each group
    reversed_indices = list(range(5, -1, -1)) + list(range(11, 5, -1))

    for idx, i in enumerate(reversed_indices):
        group_index = i // 6
        network_index = i % 6
        group = GROUPS[group_index]
        filename = f"{group}_Net{network_index + 1}.txt"
        file_path = os.path.join(DATA_DIR, filename)

        data = np.loadtxt(file_path)
        X, Y = curve(data)
        color_rgb = color[network_index]
        vertical_position = y_positions[idx]

        ax.plot(X, Y + vertical_position, color=color_rgb, linewidth=0.01)
        ax.plot(X, np.zeros_like(Y) + vertical_position,
                color=color_rgb, linewidth=0.01)
        ax.fill_between(
            X, vertical_position, Y + vertical_position,
            color=color_rgb, alpha=0.6, zorder=100 - idx
        )

    ax.grid(visible=True, axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.grid(visible=False, axis='y')

    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    yticklabels = [
        'DMN', 'FPN', 'VAN', 'DAN', 'SMN', 'VN',
        'DMN', 'FPN', 'VAN', 'DAN', 'SMN', 'VN'
    ]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(yticklabels, fontsize=8)

    x_min, x_max = ax.get_xlim()
    ax.set_xlim(-2.3, x_max)
    ax.set_xticks(np.arange(-2, np.ceil(x_max), 1))
    ax.tick_params(axis='x', labelsize=8)

    asd_center = (y_positions[0] + y_positions[5]) / 2
    control_center = (y_positions[6] + y_positions[11]) / 2

    ax.text(0.98, asd_center, 'ASD', fontsize=8, ha='right', va='center',
            transform=ax.get_yaxis_transform())
    ax.text(0.98, control_center, 'Controls', fontsize=8, ha='right', va='center',
            transform=ax.get_yaxis_transform())

    separator_y = (y_positions[5] + y_positions[6]) / 2
    ax.axhline(y=separator_y, color='gray', linestyle='--',
               linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, '7net_test.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    means = calculate_network_means()
    return compare_asd_movement(means)


if __name__ == "__main__":
    main()
