from mgdas_v2.splits import build_grouped_split_manifest, split_examples
from mgdas_v2.types import BBQExample


def make_example(identifier: str, group_id: str, attribute: str) -> BBQExample:
    return BBQExample(
        example_id=identifier,
        group_id=group_id,
        attribute=attribute,
        context="context",
        question="question",
        choices=("group a", "group b", "unknown"),
        label_index=2,
        stereotype_index=0,
        anti_stereotype_index=1,
        unknown_index=2,
        is_ambiguous=True,
    )


def test_group_variants_never_cross_pools() -> None:
    examples = []
    for group_number in range(20):
        for variant_number in range(4):
            examples.append(
                make_example(
                    f"e{group_number}-{variant_number}", f"g{group_number}", "Age"
                )
            )
    ratios = {"cal": 0.3, "search": 0.3, "cert": 0.2, "test": 0.2}
    manifest = build_grouped_split_manifest(examples, ratios, seed=42)
    pools = split_examples(examples, manifest)
    pool_by_group = {}
    for pool_name, rows in pools.items():
        for row in rows:
            pool_by_group.setdefault(row.group_id, set()).add(pool_name)
    assert all(len(pool_names) == 1 for pool_names in pool_by_group.values())
    assert all(pools[pool_name] for pool_name in pools)

