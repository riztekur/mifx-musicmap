from kedro.pipeline import Pipeline, node, pipeline

from .nodes import *

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=plot_univariate,
                inputs="processed_musicmap_human_users",
                outputs="plot_univariate",
                name="univariate_analysis"
            ),
            node(
                func=plot_correlation,
                inputs="processed_musicmap_human_users",
                outputs="plot_correlation",
                name="correlation_analysis"
            ),
            node(
                func=plot_tenure_correlation,
                inputs="processed_musicmap_human_users",
                outputs="plot_tenure_correlation",
                name="tenure_correlation_analysis"
            ),
            node(
                func=plot_adopter_vs_non_adopter,
                inputs="processed_musicmap_human_users",
                outputs="plot_adopter_vs_non_adopter",
                name="adopter_vs_non_adopter_analysis"
            ),
            node(
                func=mann_whitney_test,
                inputs="processed_musicmap_human_users",
                outputs="mann_whitney_test",
                name="mann_whitney_test"
            ),
            node(
                func=chi_square_test,
                inputs="processed_musicmap_human_users",
                outputs="chi_square_test",
                name="chi_square_test"
            ),
            node(
                func=plot_songs_vs_tenure,
                inputs="processed_musicmap_human_users",
                outputs="plot_songs_vs_tenure",
                name="songs_vs_tenure_scatter"
            )
        ]
    )