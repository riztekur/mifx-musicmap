from kedro.pipeline import Pipeline, node, pipeline

from .nodes import *

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=fill_missing_net_users,
                inputs="raw_musicmap",
                outputs="processed_musicmap_with_filled_net_users",
                name="fill_missing_net_users"
            ),
            node(
                func=make_age_category,
                inputs="processed_musicmap_with_filled_net_users",
                outputs="processed_musicmap_with_age_category",
                name="make_age_category"
            ),
            node(
                func=make_gender,
                inputs="processed_musicmap_with_age_category",
                outputs="processed_musicmap_with_gender",
                name="make_gender"
            ),
            node(
                func=fill_missing_good_country,
                inputs="processed_musicmap_with_gender",
                outputs="processed_musicmap_with_filled_good_country",
                name="fill_missing_good_country"
            ),
            node(
                func=make_consistent_friend_cnt,
                inputs="processed_musicmap_with_filled_good_country",
                outputs="processed_musicmap_with_consistent_friend_cnt",
                name="make_consistent_friend_cnt"
            ),
            node(
                func=make_songsListened_monthly,
                inputs="processed_musicmap_with_consistent_friend_cnt",
                outputs="processed_musicmap_with_songsListened_monthly",
                name="make_songsListened_monthly"
            ),
            node(
                func=make_social_activity,
                inputs="processed_musicmap_with_songsListened_monthly",
                outputs="processed_musicmap_with_social_activity",
                name="make_social_activity"
            ),
            node(
                func=split_bot_users,
                inputs="processed_musicmap_with_social_activity",
                outputs=["processed_musicmap_human_users", "processed_musicmap_bot_users"],
                name="split_bot_users"
            ),
            node(
                func=fix_missing_values,
                inputs="processed_musicmap_human_users",
                outputs="processed_musicmap_no_missing",
                name="fix_missing_values"
            ),
            node(
                func=label_encode,
                inputs="processed_musicmap_no_missing",
                outputs="processed_musicmap",
                name="label_encode"
            ),
            node(
                func=make_bins,
                inputs="processed_musicmap",
                outputs="bins",
                name="make_bins"
            ),
            node(
                func=woe_transformer,
                inputs=["processed_musicmap","bins"],
                outputs="musicmap_woe",
                name="woe_transformer"
            ),
            node(
                func=split_dataset,
                inputs="musicmap_woe",
                outputs=["train", "test"],
                name="split_dataset"
            )
        ]
    )