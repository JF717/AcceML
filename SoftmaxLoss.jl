function softmax(theta, y, mask):
    N, T, V = theta.shape
    theta_flat = theta.reshape(N*T, V)
    y_flat = y.reshape(N*T)
    mask_flat = mask.reshape(N*T)

    probs = np.exp(theta_flat - np.max(theta_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dtheta_flat = probs.copy()
    dtheta_flat[np.arange(N * T), y_flat] -= 1
    dtheta_flat /= N
    dtheta_flat *= mask_flat[:, None]

    dtheta = dtheta_flat.reshape(N, T, V)
    return loss, dtheta
