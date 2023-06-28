import os
import pandas as pd
from pandas.api.types import is_integer_dtype, is_float_dtype
from extractor.context_extractor import FeatureExtractor


def run_mip_feature_extractor(instance_path):
    # Extract feature
    feature_extractor = FeatureExtractor(problem_instance_file=instance_path)
    static_features, bipartite_graph = feature_extractor.extract_feature(), feature_extractor.extract_bipartite_graph()

    # Save static_features as a CSV file
    static_output_path = os.path.splitext(instance_path)[0] + "_static_features.csv"
    static_features.to_csv(static_output_path, index=False)
    print("Static features saved to:", static_output_path)

    # Check if static_features file exists
    assert os.path.isfile(static_output_path), "Static features file not found."
    # Check if the output files are non-empty
    assert os.path.getsize(static_output_path) > 0, "Static features file is empty."
    # Check the data types of static features
    assert is_integer_dtype(static_features['var_type']), "Invalid data type for 'var_type' in static_features."
    assert is_float_dtype(static_features[['var_lb', 'var_ub']].values), \
        "Invalid data type for 'var_lb' or 'var_ub' in static_features."
    # Check objective sense
    assert static_features['objective_sense'].iloc[0] in ['minimize', 'maximize'], "Invalid objective sense value."
    # Check the actual output
    assert isinstance(static_features, pd.DataFrame), "Invalid type for static_features."

    # Save bipartite_graph as a CSV file
    graph_output_path = os.path.splitext(instance_path)[0] + "_bipartite_graph.csv"
    bipartite_graph.to_csv(graph_output_path, index=False)
    print("Bipartite graph saved to:", graph_output_path)

    # Check if bipartite_graph file exists
    assert os.path.isfile(graph_output_path), "Bipartite graph file not found."
    # Check if the output files are non-empty
    assert os.path.getsize(graph_output_path) > 0, "Bipartite graph file is empty."
    # Check the actual output
    assert isinstance(bipartite_graph, pd.DataFrame), "Invalid type for bipartite_graph."

    # Check the data types of bipartite_graph
    assert bipartite_graph[['variable', 'constraint', 'coefficient']].dtypes.eq(object).all(), \
        "Invalid data type for 'variable', 'constraint', 'coefficient' in bipartite_graph."


if __name__ == "__main__":
    instance_path = "C:/Users/a739095/pr110447-ai_coe_pi_atlas/projects/cr/generic/PRTCJP-1345/" \
                    "data/neos-5140963-mincio.mps.gz"

    # Create MIP instance
    run_mip_feature_extractor(instance_path)
