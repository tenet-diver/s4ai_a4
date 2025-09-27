import numpy as np
from scipy.ndimage import gaussian_filter
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(1)

CHAR_NAMES = ['R', 'U', 'S']
NUM_CHARS = len(CHAR_NAMES)
NUM_VARIATIONS_PER_CHAR = 4
TOTAL_INPUTS = NUM_CHARS * NUM_VARIATIONS_PER_CHAR

characters = {
  "R": np.array([
    [1, 1, 1, 1, 0],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 0],
    [1, 0, 0, 1, 0],
    [1, 0, 0, 0, 1]
  ]),
  "U": np.array([
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [0, 1, 1, 1, 0]
  ]),
  "S": np.array([
      [0, 1, 1, 1, 1],
      [1, 0, 0, 0, 0],
      [0, 1, 1, 1, 0],
      [0, 0, 0, 0, 1],
      [1, 1, 1, 1, 0]
  ])
}

correlation_matrix = None
NN1 = None
labels = []


def create_variations():
    character_variations = {
      "R": [],
      "U": [],
      "S": []
    }
    for char_name, char_matrix in characters.items():
        emphasized_strokes_variant = char_matrix.copy()
        if char_name == 'R':
            emphasized_strokes_variant[2, :4] = 0.9
            emphasized_strokes_variant[:, 0] = 0.85
        elif char_name == 'U':
            emphasized_strokes_variant[4, 1:4] = 0.9
        elif char_name == 'S':
            emphasized_strokes_variant[2, 1:4] = 0.85

        blurred_variant = gaussian_filter(char_matrix.astype(float), sigma=0.3)
        if blurred_variant.max() > 0:
            blurred_variant = blurred_variant / blurred_variant.max()
        blurred_variant = np.clip(blurred_variant, 0, 1)

        noisy_variant = char_matrix.copy()
        noise_mask = np.random.random((5, 5)) < 0.15
        noisy_variant[noise_mask] = 1 - noisy_variant[noise_mask]

        combined_variant = char_matrix.copy().astype(float)
        combined_noise_mask = np.random.random((5, 5)) < 0.1
        combined_variant[combined_noise_mask] = 1 - combined_variant[combined_noise_mask]
        combined_variant = gaussian_filter(combined_variant, sigma=0.2)
        if combined_variant.max() > 0:
            combined_variant = combined_variant / combined_variant.max()
        combined_variant = np.clip(combined_variant, 0, 1)

        character_variations[char_name] = [emphasized_strokes_variant, blurred_variant, noisy_variant, combined_variant]

    return character_variations


def create_inputs(variations):
    flattened_inputs = []
    for char_name in CHAR_NAMES:
        for variation_idx, variation_matrix in enumerate(variations[char_name]):
            flattened_inputs.append(variation_matrix.flatten())
            labels.append(f"{char_name}_v{variation_idx+1}")
    return flattened_inputs


    

def compute_correlation_matrix(inputs):
    global correlation_matrix
    correlation_matrix = np.zeros((TOTAL_INPUTS, TOTAL_INPUTS))

    for row_idx in range(TOTAL_INPUTS):
        for col_idx in range(TOTAL_INPUTS):
            correlation_matrix[row_idx, col_idx] = np.dot(inputs[row_idx], inputs[col_idx])

    return correlation_matrix


def analyze_correlation():
    char_confusion = np.zeros((NUM_CHARS, NUM_CHARS))

    for char_idx_1 in range(NUM_CHARS):
        for char_idx_2 in range(NUM_CHARS):
            correlation_values = []
            for var_idx_1 in range(NUM_VARIATIONS_PER_CHAR):
                for var_idx_2 in range(NUM_VARIATIONS_PER_CHAR):
                    matrix_row = char_idx_1 * NUM_VARIATIONS_PER_CHAR + var_idx_1
                    matrix_col = char_idx_2 * NUM_VARIATIONS_PER_CHAR + var_idx_2
                    correlation_values.append(correlation_matrix[matrix_row, matrix_col])
            char_confusion[char_idx_1, char_idx_2] = np.mean(correlation_values)

    print("\n📊 Character Confusion Matrix:")
    header = "    " + "".join(f"{c:7s}" for c in CHAR_NAMES)
    print(header)
    for char_idx, char_name in enumerate(CHAR_NAMES):
        row_text = f"{char_name}: " + "".join(
            f"{char_confusion[char_idx, col_idx]:6.2f} "
            for col_idx in range(NUM_CHARS)
        )
        print(row_text)


def create_nn1(variations):
    global NN1
    templates = []

    for char_name in CHAR_NAMES:
        char_avg = np.mean([var.flatten() for var in variations[char_name]], axis=0)
        templates.append(char_avg)

    NN1 = np.array(templates)
    for template_idx in range(NUM_CHARS):
        NN1[template_idx] = NN1[template_idx] / np.linalg.norm(NN1[template_idx])

    return NN1


def test_nn1(input_vec):
    if len(input_vec.shape) > 1:
        input_vec = input_vec.flatten()
    scores = np.dot(NN1, input_vec)
    return scores


def test_edge_cases():
    print("\n🔍 Testing edge cases:")

    all_ones = np.ones(25)
    scores_ones = test_nn1(all_ones)
    print(f"All ones: R={scores_ones[0]:.3f}, U={scores_ones[1]:.3f}, S={scores_ones[2]:.3f}")
    print(f"  → Predicted: {CHAR_NAMES[np.argmax(scores_ones)]}")

    all_zeros = np.zeros(25)
    scores_zeros = test_nn1(all_zeros)
    print(f"All zeros: R={scores_zeros[0]:.3f}, U={scores_zeros[1]:.3f}, S={scores_zeros[2]:.3f}")

    print("\n⚠️  Testing single-pixel perturbations...")
    misclassified_count = 0

    for char_name in CHAR_NAMES:
        char_matrix = characters[char_name]
        for i in range(5):
            for j in range(5):
                test_input = char_matrix.copy()
                test_input[i, j] = 1 - test_input[i, j]
                scores = test_nn1(test_input)
                true_idx = CHAR_NAMES.index(char_name)

                if np.argmax(scores) != true_idx:
                    misclassified_count += 1
                    if misclassified_count == 1:
                        predicted_char = CHAR_NAMES[np.argmax(scores)]
                        print(f"  Example: {char_name} → {predicted_char} when flipping pixel [{i},{j}]")

    print(f"  Found {misclassified_count} misclassifications with single-pixel changes")

    print("\n⚠️  Testing two-pixel perturbations...")
    two_pixel_misclassified = 0

    for char_name in CHAR_NAMES:
        char_matrix = characters[char_name]
        for i1 in range(5):
            for j1 in range(5):
                for i2 in range(i1, 5):
                    for j2 in range(5):
                        if i1 == i2 and j2 <= j1:
                            continue

                        test_input = char_matrix.copy()
                        test_input[i1, j1] = 1 - test_input[i1, j1]
                        test_input[i2, j2] = 1 - test_input[i2, j2]
                        scores = test_nn1(test_input)
                        true_idx = CHAR_NAMES.index(char_name)

                        if np.argmax(scores) != true_idx:
                            two_pixel_misclassified += 1
                            if two_pixel_misclassified == 1:
                                predicted_char = CHAR_NAMES[np.argmax(scores)]
                                print(f"  Example: {char_name} → {predicted_char} when flipping pixels [{i1},{j1}] and [{i2},{j2}]")

    print(f"  Found {two_pixel_misclassified} misclassifications with two-pixel changes")

    return scores_ones, scores_zeros, misclassified_count, two_pixel_misclassified


def find_undecidable_inputs():
    print("\n🤔 Finding undecidable inputs using null-space method...")

    constraint_matrix = np.vstack([
        # difference between R & U
        NN1[0] - NN1[1],
        # difference between U & S
        NN1[0] - NN1[2]
    ])

    _, _, V = np.linalg.svd(constraint_matrix)
    null_space = V[2:].T

    coeffs = np.random.randn(null_space.shape[1])
    undecidable_input = null_space @ coeffs

    undecidable_input = undecidable_input / np.max(np.abs(undecidable_input)) * 0.5 + 0.5
    undecidable_input = np.clip(undecidable_input, 0, 1)

    scores = test_nn1(undecidable_input)
    score_std = np.std(scores)

    print(f"  Undecidable input scores: R={scores[0]:.3f}, U={scores[1]:.3f}, S={scores[2]:.3f}")
    print(f"  Score standard deviation: {score_std:.6f} (close to 0 = undecidable)")

    return undecidable_input, scores, score_std


def visualize_results(variations, inputs):
    fig, axes = plt.subplots(NUM_CHARS, NUM_VARIATIONS_PER_CHAR + 1, figsize=(12, 8))
    for char_idx, char_name in enumerate(CHAR_NAMES):
        axes[char_idx, 0].imshow(characters[char_name], cmap='gray', vmin=0, vmax=1)
        axes[char_idx, 0].set_title(f'{char_name} (Original)')
        axes[char_idx, 0].axis('off')

        for var_idx, variation_matrix in enumerate(variations[char_name]):
            axes[char_idx, var_idx+1].imshow(variation_matrix, cmap='gray', vmin=0, vmax=1)
            axes[char_idx, var_idx+1].set_title(f'Var {var_idx+1}')
            axes[char_idx, var_idx+1].axis('off')

    plt.suptitle('Characters and Variations')
    plt.tight_layout()
    plt.savefig('characters.png')

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.1f',
                xticklabels=labels, yticklabels=labels,
                cmap='coolwarm', center=0, ax=ax)
    ax.set_title(f'Correlation Matrix ({TOTAL_INPUTS}x{TOTAL_INPUTS})')
    plt.tight_layout()
    plt.savefig('correlation.png')

    classification_scores = []
    for input_vector in inputs:
        classification_scores.append(test_nn1(input_vector))
    classification_scores = np.array(classification_scores)

    fig, ax = plt.subplots(figsize=(12, 6))
    x_positions = np.arange(len(labels))
    bar_width = 0.25

    ax.bar(x_positions - bar_width, classification_scores[:, 0], bar_width, label='R', color='blue')
    ax.bar(x_positions, classification_scores[:, 1], bar_width, label='U', color='green')
    ax.bar(x_positions + bar_width, classification_scores[:, 2], bar_width, label='S', color='red')

    ax.axhline(y=2.5, color='black', linestyle='--', alpha=0.5, label='Threshold')
    ax.set_xlabel('Input')
    ax.set_ylabel('Score')
    ax.set_title('NN1 Classification Scores')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('scores.png')

    return classification_scores


if __name__ == "__main__":
    character_variations = create_variations()
    flattened_inputs = create_inputs(character_variations)
    correlation_matrix = compute_correlation_matrix(flattened_inputs)
    analyze_correlation()
    neural_network_matrix = create_nn1(character_variations)
    classification_scores = visualize_results(character_variations, flattened_inputs)

    num_correct_predictions = 0
    for label_idx, label_text in enumerate(labels):
        true_character = label_text[0]
        true_class_index = CHAR_NAMES.index(true_character)
        predicted_class_index = np.argmax(classification_scores[label_idx])
        if true_class_index == predicted_class_index:
            num_correct_predictions += 1

    edge_case_results = test_edge_cases()
    undecidable_results = find_undecidable_inputs()

    accuracy_percentage = 100 * num_correct_predictions / len(labels)
    print(f"\n✅ Accuracy: {num_correct_predictions}/{len(labels)} = {accuracy_percentage:.1f}%")
    print("Saved: characters.png, correlation.png, scores.png")

    if len(edge_case_results) > 3 and edge_case_results[2] > 0:
        print(f"⚠️  {edge_case_results[2]} single-pixel vulnerabilities found")
    else:
        print("✅ No single-pixel vulnerabilities found")

    if len(edge_case_results) > 3 and edge_case_results[3] > 0:
        print(f"⚠️  {edge_case_results[3]} two-pixel vulnerabilities found")
    else:
        print("✅ No two-pixel vulnerabilities found")

    print(f"🤔 Undecidable input std: {undecidable_results[2]:.6f}")