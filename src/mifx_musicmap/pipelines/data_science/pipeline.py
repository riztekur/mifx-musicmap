from kedro.pipeline import Pipeline, node, pipeline

from .nodes import *

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=user_clustering,
                inputs="processed_musicmap_no_missing",
                outputs="user_clustering",
                name="user_cluster",
            ),
            node(
                func=train_model,
                inputs="train",
                outputs="model",
                name="train_model",
            ),
            node(
                func=evaluate_model,
                inputs=["model", "bins", "train", "test"],
                outputs=["performance_summary", "train_pic", "test_pic", "card_df"],
                name="evaluate_model",
            ),
            node(
                func=train_tree_model,
                inputs="processed_musicmap",
                outputs=["tree", "tree_fig"],
                name="train_tree_model",
            )
        ]
    )