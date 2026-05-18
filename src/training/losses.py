def binary_cross_entropy_with_logits(y_pred, y_true):
    return torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y_true)

def focal_loss(y_pred, y_true, alpha=1.0, gamma=2.0):
    bce_loss = binary_cross_entropy_with_logits(y_pred, y_true)
    p_t = torch.exp(-bce_loss)
    focal_loss = alpha * (1 - p_t) ** gamma * bce_loss
    return focal_loss.mean()

def custom_loss(y_pred, y_true):
    # Example of a custom loss function that combines multiple losses
    bce = binary_cross_entropy_with_logits(y_pred, y_true)
    focal = focal_loss(y_pred, y_true)
    return bce + focal

def get_loss_function(loss_type='binary_cross_entropy'):
    if loss_type == 'binary_cross_entropy':
        return binary_cross_entropy_with_logits
    elif loss_type == 'focal_loss':
        return focal_loss
    elif loss_type == 'custom':
        return custom_loss
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")