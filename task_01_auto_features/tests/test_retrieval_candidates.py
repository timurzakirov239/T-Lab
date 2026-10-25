import unittest

from task_01_auto_features.src.retrieval_candidates import (
    AFFINITY_QUOTAS,
    BASELINE_QUOTAS,
    REPEAT_HEAVY_QUOTAS,
    RetrievalCandidateGenerator,
)
from task_01_auto_features.src.run_pilot import CandidateGenerator as P001CandidateGenerator


class RetrievalCandidateTests(unittest.TestCase):
    @staticmethod
    def event(uid, item, timestamp, event_type="listen"):
        return {"uid": uid, "item_id": item, "timestamp": timestamp, "event_type": event_type}

    def test_baseline_is_exactly_equivalent_to_p001_at_k100(self):
        artists = {item: (item % 40, item % 40) for item in range(1, 241)}
        albums = {item: (item % 30,) for item in range(1, 241)}
        config = {"max_candidates": 100, "candidate_quotas": BASELINE_QUOTAS.copy()}
        p001 = P001CandidateGenerator(artists, albums, config)
        p002 = RetrievalCandidateGenerator(artists, albums)
        for item in range(1, 241):
            event = self.event(9, item, item)
            p001.observe(event)
            p002.observe(event)
        expected = p001.candidates(9, 241)
        result = p002.candidates(9, 241, "baseline")
        self.assertEqual(result.candidate_items, expected)
        self.assertEqual(len(result.candidate_items), 100)
        self.assertEqual(len(result.source_pools["popular"]), 200)
        self.assertEqual(set(result.quota_contributions), {"popular", "repeat", "artist", "album"})
        self.assertTrue(all(result.quota_contributions[k] <= BASELINE_QUOTAS[k] for k in BASELINE_QUOTAS))
        self.assertEqual({k: len(v) for k, v in result.selected_by_source.items()}, result.quota_contributions)
        self.assertEqual(result.fallback_contributions["popular_fallback"], len(result.fallback_items))
        self.assertTrue(all(result.selection_sources[item] == "popular_fallback" for item in result.fallback_items))
        self.assertEqual(set(result.selection_sources), set(result.candidate_items))

    def test_prefix_causality_and_equal_timestamp_safety(self):
        def run(future_variant):
            generator = RetrievalCandidateGenerator({}, {}, max_candidates=10)
            generator.observe(self.event(1, 4, 10, "listen"))
            generator.observe(self.event(1, 6, 10, "unlike"))
            result = generator.candidates(1, 11, "baseline")
            # Future fields are deliberately changed but never passed to observe.
            _future = self.event(1, future_variant, 20, "listen")
            return result

        left = run(100)
        right = run(999)
        self.assertEqual(left.candidate_items, right.candidate_items)
        self.assertEqual(left.source_pools, right.source_pools)
        self.assertEqual(left.provenance, right.provenance)
        generator = RetrievalCandidateGenerator({}, {})
        generator.observe(self.event(1, 1, 10))
        with self.assertRaisesRegex(ValueError, "strictly before"):
            generator.candidates(1, 10)
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            generator.observe(self.event(1, 2, 9))

    def test_deterministic_ties_dedup_quotas_provenance_and_overlaps(self):
        # Duplicate mapping IDs are normalized, and all one-listen popularity ties
        # resolve by ascending item_id in every source rank.
        artists = {1: (7, 7), 2: (7,), 3: (8,)}
        albums = {1: (4,), 2: (4, 4), 3: (5,)}
        generator = RetrievalCandidateGenerator(
            artists, albums, max_candidates=3,
            candidate_quotas={"popular": 1, "repeat": 1, "artist": 1, "album": 1},
        )
        for item in (1, 2, 3):
            generator.observe(self.event(1, item, item))
        result = generator.candidates(1, 4)
        self.assertEqual(result.source_pools["popular"], [1, 2, 3])
        self.assertEqual(result.candidate_items, [1, 3, 2])
        self.assertEqual(result.quota_contributions, {"popular": 1, "repeat": 1, "artist": 1, "album": 0})
        self.assertEqual(result.selected_by_source["popular"], [1])
        self.assertEqual(result.selected_by_source["repeat"], [3])
        self.assertEqual(result.selection_sources[2], "artist")
        self.assertEqual(result.overlaps["popular&repeat"], 3)
        self.assertEqual(result.provenance[1], ["popular", "repeat", "artist", "album"])
        self.assertEqual(result.overlaps["union_items"], 3)
        self.assertEqual(len(result.candidate_items), len(set(result.candidate_items)))

    def test_profile_depth_variants_and_pool_cache_invalidation(self):
        artists = {item: (item,) for item in range(1, 41)}
        generator = RetrievalCandidateGenerator(artists, {}, max_candidates=100)
        for item in range(1, 41):
            generator.observe(self.event(1, item, item))
        baseline = generator.source_pools(1, 41, "baseline")
        affinity = generator.source_pools(1, 41, "affinity")
        self.assertNotIn(20, baseline["artist"])
        self.assertIn(20, affinity["artist"])
        cached = generator._pool_cache
        generator.observe(self.event(1, 41, 40, "unlike"))
        self.assertIsNone(generator._pool_cache_key)
        refreshed = generator.candidates(1, 42, "baseline")
        self.assertIsNot(cached, generator._pool_cache)
        self.assertIn(41, generator.catalog)
        self.assertNotIn(41, refreshed.candidate_items)  # Unlike adds catalog only.

    def test_quota_variants_use_requested_order_and_rff_has_no_quotas(self):
        generator = RetrievalCandidateGenerator({}, {}, max_candidates=100)
        for item in range(1, 201):
            generator.observe(self.event(1, item, item))
        baseline = generator.candidates(1, 201, "baseline")
        repeat_heavy = generator.candidates(1, 201, "repeat_heavy")
        affinity = generator.candidates(1, 201, "affinity")
        rrf = generator.candidates(1, 201, "rank_fusion")
        self.assertEqual(BASELINE_QUOTAS, {"popular": 25, "repeat": 25, "artist": 35, "album": 15})
        self.assertEqual(REPEAT_HEAVY_QUOTAS, {"popular": 10, "repeat": 60, "artist": 20, "album": 10})
        self.assertEqual(AFFINITY_QUOTAS, {"popular": 10, "repeat": 20, "artist": 50, "album": 20})
        self.assertEqual(len(baseline.candidate_items), 100)
        self.assertEqual(len(repeat_heavy.candidate_items), 100)
        self.assertEqual(len(affinity.candidate_items), 100)
        self.assertEqual(baseline.quota_contributions, {"popular": 25, "repeat": 25, "artist": 0, "album": 0})
        self.assertEqual(repeat_heavy.quota_contributions, {"popular": 10, "repeat": 60, "artist": 0, "album": 0})
        self.assertEqual(affinity.quota_contributions, {"popular": 10, "repeat": 20, "artist": 0, "album": 0})
        self.assertEqual(rrf.quota_contributions, {})
        self.assertEqual(rrf.selected_by_source, {})
        self.assertTrue(all(source is None for source in rrf.selection_sources.values()))
        self.assertEqual(rrf.fallback_items, [])
        self.assertEqual(rrf.fallback_contributions, {})

    def test_rrf_matches_hand_computed_example_and_ties(self):
        pools = {
            "popular": [1, 2],
            "repeat": [2, 3],
            "artist": [3, 1],
            "album": [4],
        }
        scores = RetrievalCandidateGenerator._rrf(pools, c=60)
        self.assertAlmostEqual(scores[1], 1 / 61 + 1 / 62)
        self.assertAlmostEqual(scores[2], 1 / 62 + 1 / 61)
        self.assertAlmostEqual(scores[3], 1 / 62 + 1 / 61)
        ranked = sorted(scores, key=lambda item: (-scores[item], item))
        self.assertEqual(ranked, [1, 2, 3, 4])

    def test_rrf_result_keeps_pool_provenance_and_seen_catalog_only(self):
        generator = RetrievalCandidateGenerator({}, {}, max_candidates=2, source_depth=2)
        generator.observe(self.event(1, 7, 1, "listen"))
        generator.observe(self.event(1, 9, 2, "listen"))
        result = generator.candidates(1, 3, "rank_fusion")
        self.assertTrue(set(result.candidate_items).issubset({7, 9}))
        self.assertTrue(all(item in generator.catalog for item in result.provenance))
        self.assertEqual(set(result.source_pools), {"popular", "repeat", "artist", "album"})
        self.assertFalse(hasattr(result, "target_item"))


if __name__ == "__main__":
    unittest.main()
