import numpy as np

def compute_anomaly_score(predictions, threshold=0.5):
    """
    Compute the anomaly score for a given set of predictions.

    Parameters:
    - predictions: np.ndarray, predicted classes for each beat (0 for normal, 1 for SVEB, 2 for VEB)
    - threshold: float, threshold for determining anomalies

    Returns:
    - score: float, computed anomaly score
    """
    k = len(predictions)
    score = (1 / k) * np.sum(predictions != 0)
    return score

def evaluate_anomaly_score(predictions, duration_threshold, min_duration):
    """
    Evaluate if the anomaly score meets the clinical event detection criteria.

    Parameters:
    - predictions: np.ndarray, predicted classes for each beat
    - duration_threshold: float, threshold for anomaly score
    - min_duration: int, minimum duration for sustained events

    Returns:
    - event_detected: bool, True if an event is detected, False otherwise
    """
    score = compute_anomaly_score(predictions)
    sustained_event = (score >= duration_threshold) and (len(predictions) >= min_duration)
    return sustained_event

def main():
    # Example usage
    predictions = np.array([0, 1, 0, 2, 0, 1, 1, 0])  # Example predictions
    threshold = 0.5
    min_duration = 5

    score = compute_anomaly_score(predictions)
    event_detected = evaluate_anomaly_score(predictions, threshold, min_duration)

    print(f"Anomaly Score: {score}")
    print(f"Event Detected: {event_detected}")

if __name__ == "__main__":
    main()