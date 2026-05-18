def detect_events(anomaly_scores, threshold, min_duration):
    """
    Detect clinical events based on anomaly scores.

    Parameters:
    anomaly_scores (list): List of anomaly scores for each window.
    threshold (float): Threshold for detecting an event.
    min_duration (int): Minimum duration (in number of windows) for an event to be considered valid.

    Returns:
    list: List of detected events, where each event is represented by its start and end indices.
    """
    events = []
    event_start = None

    for i, score in enumerate(anomaly_scores):
        if score >= threshold:
            if event_start is None:
                event_start = i  # Start of a new event
        else:
            if event_start is not None:
                # End of the current event
                if (i - event_start) >= min_duration:
                    events.append((event_start, i - 1))
                event_start = None  # Reset event start

    # Check if there is an ongoing event at the end of the list
    if event_start is not None and (len(anomaly_scores) - event_start) >= min_duration:
        events.append((event_start, len(anomaly_scores) - 1))

    return events

def main():
    # Example usage
    anomaly_scores = [0, 0, 1, 1, 0, 1, 1, 1, 0, 0]  # Example anomaly scores
    threshold = 0.5
    min_duration = 2

    detected_events = detect_events(anomaly_scores, threshold, min_duration)
    print("Detected Events:", detected_events)

if __name__ == "__main__":
    main()