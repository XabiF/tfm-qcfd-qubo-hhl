import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error

def fit_log(x, y):
    x = np.array(x)
    y = np.array(y)

    # --- Linear Fit ---
    lin_params = np.polyfit(x, y, 1)
    lin_fit = np.poly1d(lin_params)
    y_lin = lin_fit(x)
    a1, b1 = lin_params
    print(f"Linear Fit:   y = {a1:.4f}x + {b1:.4f}")

    # --- Quadratic Fit ---
    quad_params = np.polyfit(x, y, 2)
    quad_fit = np.poly1d(quad_params)
    y_quad = quad_fit(x)
    a2, b2, c2 = quad_params
    print(f"Quadratic Fit: y = {a2:.4f}x² + {b2:.4f}x + {c2:.4f}")

    # --- Cubic Fit ---
    cubic_params = np.polyfit(x, y, 3)
    cubic_fit = np.poly1d(cubic_params)
    y_cubic = cubic_fit(x)
    a3, b3, c3, d3 = cubic_params
    print(f"Cubic Fit:    y = {a3:.4f}x³ + {b3:.4f}x² + {c3:.4f}x + {d3:.4f}")

    # --- Quartic Fit ---
    quartic_params = np.polyfit(x, y, 4)
    quartic_fit = np.poly1d(quartic_params)
    y_quartic = quartic_fit(x)
    a4, b4, c4, d4, e4 = quartic_params
    print(f"Quartic Fit:  y = {a4:.4f}x⁴ + {b4:.4f}x³ + {c4:.4f}x² + {d4:.4f}x + {e4:.4f}")

    # --- Quintic Fit ---
    quintic_params = np.polyfit(x, y, 5)
    quintic_fit = np.poly1d(quintic_params)
    y_quintic = quintic_fit(x)
    a5, b5, c5, d5, e5, f5 = quintic_params
    print(f"Quintic Fit:  y = {a5:.4f}x⁵ + {b5:.4f}x⁴ + {c5:.4f}x³ + {d5:.4f}x² + {e5:.4f}x + {f5:.4f}")

    # --- Exponential Fit ---
    def exponential(x, a, b):
        return a * np.exp(b * x)

    try:
        x_scaled = x / np.max(x)  # scale to avoid overflow
        exp_params, _ = curve_fit(
            exponential, x_scaled, y,
            p0=(max(y), 0.001), maxfev=10000
        )
        y_exp = exponential(x_scaled, *exp_params)
        a_exp, b_exp = exp_params
        print("Exponential Fit (x scaled):")
        print(f"y = {a_exp:.4f} * e^({b_exp:.4f} * (x / max(x)))")
    except Exception as e:
        print(f"\nExponential Fit failed: {e}")
        y_exp = np.full_like(y, np.nan)

    # --- Scores ---
    fits = {
        "Linear": y_lin, "Quadratic": y_quad, "Cubic": y_cubic,
        "Quartic": y_quartic, "Quintic": y_quintic, "Exponential": y_exp
    }

    for name, y_pred in fits.items():
        try:
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            print(f"{name:10s}: R² = {r2:.4f}, MSE = {mse:.4f}")
        except Exception:
            print(f"{name:10s}: scoring failed (probably NaNs)")
