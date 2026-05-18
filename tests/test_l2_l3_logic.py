import pytest
from src.inference.l2_score import compute_anomaly_score
from src.inference.l3_events import detect_events

def test_anomaly_score():
    # Test case for anomaly score calculation
    y_hat = [0, 1, 1, 0, 2, 0, 1]  # Example predictions
    expected_score = (1/7) * sum(1 for y in y_hat if y != 0)  # S(Wt) calculation
    score = compute_anomaly_score(y_hat)
    assert score == expected_score, f"Expected {expected_score}, but got {score}"

def test_event_detection():
    # Test case for event detection logic
    anomaly_scores = [0.1, 0.6, 0.7, 0.2, 0.8]  # Example anomaly scores
    threshold = 0.5
    min_duration = 2
    events = detect_events(anomaly_scores, threshold, min_duration)
    expected_events = [(1, 3)]  # Example expected events based on the scores
    assert events == expected_events, f"Expected {expected_events}, but got {events}"