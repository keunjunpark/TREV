# Notebooks Directory

This directory contains Jupyter notebooks for experiments, plotting, and analysis.

## Important Note: Naming Convention

⚠️ **The notebooks in this directory use the old naming convention for measurement methods.**

### Old vs New Naming

The TREV library has updated its measurement method names. If you're reading these notebooks, please note the following mappings:

| **Old Name (used in notebooks)** | **New Name (current library)** |
|----------------------------------|-------------------------------|
| `MeasureMethod.SAMPLING` | `MeasureMethod.PERFECT_SAMPLING` |
| `MeasureMethod.CORRECT_SAMPLING` | `MeasureMethod.RIGHT_SUFFIX_SAMPLING` |
| `MeasureMethod.CONTRACTION` | `MeasureMethod.FULL_CONTRACTION` |
| `MeasureMethod.EFFICIENT_CONTRACTION` | `MeasureMethod.EFFICIENT_CONTRACTION` *(unchanged)* |

### Using the Notebooks

When running these notebooks with the current version of TREV, you may need to update the method names to match the new convention. For example:

**Old code in notebooks:**
```python
exp_val = circuit.get_expectation_value(theta, hamiltonian, MeasureMethod.SAMPLING, shots=10000)
```

**Updated code for current library:**
```python
exp_val = circuit.get_expectation_value(theta, hamiltonian, MeasureMethod.PERFECT_SAMPLING, shots=10000)
```

### Why the Change?

The naming was updated to better reflect what each method actually does:
- **Perfect Sampling**: Exact sampling from the quantum state probability distribution
- **Right Suffix Sampling**: Advanced sampling method using right suffix tensor contractions
- **Full Contraction**: Complete tensor network contraction for exact results
- **Efficient Contraction**: Partial tensor contraction balancing accuracy and efficiency

### Notebooks Overview

- **`create_plot_script/`**: Scripts for generating plots and visualizations from experimental data
- **`data/`**: Experimental results and benchmark data
- **`plots/`**: Generated plots and figures
- **`running_code/`**: Active experiment notebooks

For the most up-to-date usage examples with the new naming convention, please refer to the main [README.md](../README.md) in the repository root.
