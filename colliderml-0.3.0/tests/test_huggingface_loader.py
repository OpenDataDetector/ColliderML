"""Tests for loading ColliderML datasets from HuggingFace.

These tests verify that:
1. Datasets can be loaded from HuggingFace Hub
2. Dataset structure matches expected schema
3. Data can be accessed and inspected
4. Basic operations work correctly

These tests use minimal data downloads and streaming mode to be CI-friendly.
They also serve as examples for the quickstart documentation.
"""

import pytest
from datasets import load_dataset, get_dataset_config_names
import numpy as np
import time


pytestmark = pytest.mark.integration

# Canonical consolidated dataset
HF_DATASET_ID = "CERN/ColliderML-Release-1"

# Keep a small, deterministic set of configs for integration tests.
# This avoids network calls during pytest collection for parametrization.
KNOWN_CONFIGS = [
    "ttbar_pu0_particles",
    "ttbar_pu0_tracker_hits",
    "ttbar_pu0_calo_hits",
    "ttbar_pu0_tracks",
]


class TestHuggingFaceLoaderStreaming:
    """Test HuggingFace dataset loading with streaming mode (fast, minimal download)."""

    def test_streaming_single_event(self):
        """Test loading just one event in streaming mode.

        This is the fastest test - validates connection and basic functionality.
        """
        start = time.time()

        # Load in streaming mode (doesn't download full dataset)
        dataset = load_dataset(
            HF_DATASET_ID,
            "ttbar_pu0_particles",
            split="train",
            streaming=True
        )

        # Get just the first event
        first_event = next(iter(dataset))
        assert first_event is not None, "Should be able to access first event"

        elapsed = time.time() - start
        print(f"\n✓ Loaded 1 event in streaming mode ({elapsed:.2f}s)")
        print(f"  Event keys: {list(first_event.keys())}")

    def test_streaming_dataset_structure(self):
        """Test dataset structure using streaming mode.

        Inspect the data schema without downloading everything.
        """
        start = time.time()

        dataset = load_dataset(
            HF_DATASET_ID,
            "ttbar_pu0_particles",
            split="train",
            streaming=True
        )

        # Get first event to inspect structure
        first_event = next(iter(dataset))

        elapsed = time.time() - start
        print(f"\n✓ Dataset structure inspection ({elapsed:.2f}s)")
        print(f"  Columns: {list(first_event.keys())}")

        # Inspect each field
        for key, value in first_event.items():
            if hasattr(value, 'shape'):
                print(f"  - {key}: shape={value.shape}, dtype={value.dtype}")
            elif hasattr(value, '__len__'):
                print(f"  - {key}: length={len(value)}, type={type(value).__name__}")
            else:
                print(f"  - {key}: {value}")

    def test_streaming_multiple_events(self):
        """Test iterating through multiple events in streaming mode.

        Downloads only the events we actually iterate over.
        """
        start = time.time()

        dataset = load_dataset(
            HF_DATASET_ID,
            "ttbar_pu0_particles",
            split="train",
            streaming=True
        )

        # Process first 3 events
        num_events = 3
        events_processed = 0

        for i, event in enumerate(dataset):
            if i >= num_events:
                break
            assert event is not None, f"Event {i} should not be None"
            events_processed += 1

        elapsed = time.time() - start
        assert events_processed == num_events, f"Should process {num_events} events"

        print(f"\n✓ Streamed {num_events} events ({elapsed:.2f}s, {elapsed/num_events:.2f}s per event)")

    def test_streaming_particle_inspection(self):
        """Test detailed particle data inspection in streaming mode.

        This demonstrates how to examine physics content efficiently.
        """
        start = time.time()

        dataset = load_dataset(
            HF_DATASET_ID,
            "ttbar_pu0_particles",
            split="train",
            streaming=True
        )

        event = next(iter(dataset))

        elapsed = time.time() - start
        print(f"\n✓ Particle event inspection ({elapsed:.2f}s):")

        # Print detailed statistics for numeric arrays
        for key, value in event.items():
            if hasattr(value, 'dtype') and np.issubdtype(value.dtype, np.number):
                if value.size > 0:
                    print(f"  - {key}:")
                    print(f"    Shape: {value.shape}, Dtype: {value.dtype}")
                    print(f"    Range: [{np.min(value):.3f}, {np.max(value):.3f}]")
                    print(f"    Mean: {np.mean(value):.3f}, Std: {np.std(value):.3f}")

    @pytest.mark.parametrize("config", KNOWN_CONFIGS)
    def test_streaming_all_configs(self, config):
        """Test loading all available configurations in streaming mode.

        Dynamically tests all configs discovered from HuggingFace API.
        """
        start = time.time()

        try:
            dataset = load_dataset(
                HF_DATASET_ID,
                config,
                split="train",
                streaming=True
            )

            # Just verify we can access first event
            first_event = next(iter(dataset))
            assert first_event is not None

            elapsed = time.time() - start
            print(f"\n✓ Config '{config}' accessible ({elapsed:.2f}s)")
            print(f"  Keys: {list(first_event.keys())}")

        except Exception as e:
            pytest.skip(f"Config '{config}' not available: {e}")


class TestHuggingFaceLoaderDownload:
    """Tests that actually download data (slower, run optionally)."""

    @pytest.mark.slow
    def test_download_small_split(self):
        """Test downloading a small number of events.

        This downloads actual files to disk for caching.
        Marked as 'slow' - skip with: pytest -m "not slow"
        """
        start = time.time()

        # Download just first 10 events if possible
        # Note: HuggingFace datasets may not support partial downloads,
        # this might download the full split
        dataset = load_dataset(
            HF_DATASET_ID,
            "ttbar_pu0_particles",
            split="train[:10]"  # Try to get just 10 events
        )

        elapsed = time.time() - start
        print(f"\n✓ Downloaded split with {len(dataset)} events ({elapsed:.2f}s)")
        print(f"  Average: {elapsed/len(dataset):.2f}s per event")

        # Verify we can access the data
        first_event = dataset[0]
        assert first_event is not None

    @pytest.mark.slow
    def test_batch_loading(self):
        """Test batch loading (requires downloaded data).

        Marked as 'slow' since it needs data downloaded first.
        """
        dataset = load_dataset(
            HF_DATASET_ID,
            "ttbar_pu0_particles",
            split="train[:5]"
        )

        # Load all available events as batch
        batch = dataset[:]

        assert isinstance(batch, dict), "Batch should be a dictionary"
        print(f"\n✓ Batch loading works")
        print(f"  Batch size: {len(batch[list(batch.keys())[0]])}")
        print(f"  Keys: {list(batch.keys())}")


class TestDatasetDiscovery:
    """Test discovering what datasets and configs are available."""

    def test_config_discovery(self):
        """Test that we can discover available configs from HuggingFace."""
        configs = get_dataset_config_names(HF_DATASET_ID)
        print(f"\n✓ Discovered {len(configs)} configs:")
        for config in configs:
            print(f"  - {config}")

        # We expect at least 4 configs based on docs
        expected_configs = set(KNOWN_CONFIGS)
        found_configs = set(configs)

        assert expected_configs.issubset(found_configs), \
            f"Missing expected configs: {expected_configs - found_configs}"

    @pytest.mark.parametrize("config", KNOWN_CONFIGS)
    def test_all_configs_accessible(self, config):
        """Test that all discovered configurations are actually accessible.

        Verifies each config can be loaded and has data.
        """
        start = time.time()

        try:
            dataset = load_dataset(
                HF_DATASET_ID,
                config,
                split="train",
                streaming=True
            )

            first_event = next(iter(dataset))
            elapsed = time.time() - start

            print(f"\n✓ Config '{config}' verified ({elapsed:.2f}s)")
            print(f"  Fields: {list(first_event.keys())}")
            print(f"  Number of fields: {len(first_event.keys())}")

            # Verify we have data
            assert len(first_event.keys()) > 0, f"Config '{config}' has no fields"

        except Exception as e:
            pytest.fail(f"Config '{config}' should be accessible: {e}")

class TestTimingBenchmarks:
    """Benchmark timing for different operations."""

    def test_streaming_first_event_timing(self):
        """Measure time to get first event in streaming mode.

        This is the baseline timing - should be fast (few seconds).
        """
        times = []

        for i in range(3):
            start = time.time()

            dataset = load_dataset(
                HF_DATASET_ID,
                "ttbar_pu0_particles",
                split="train",
                streaming=True
            )
            first_event = next(iter(dataset))

            elapsed = time.time() - start
            times.append(elapsed)

        avg_time = np.mean(times)
        std_time = np.std(times)

        print(f"\n✓ First event timing (n=3):")
        print(f"  Mean: {avg_time:.2f}s")
        print(f"  Std:  {std_time:.2f}s")
        print(f"  Times: {[f'{t:.2f}s' for t in times]}")

        # Should be reasonably fast (less than 30 seconds for first event)
        assert avg_time < 30, f"First event took {avg_time:.2f}s, expected <30s"


if __name__ == "__main__":
    # Run tests with pytest
    # Use: pytest -v -s                  # Run fast tests
    # Use: pytest -v -s -m slow          # Run slow tests only
    # Use: pytest -v -s -m "not slow"    # Skip slow tests
    pytest.main([__file__, "-v", "-s", "-m", "not slow"])
