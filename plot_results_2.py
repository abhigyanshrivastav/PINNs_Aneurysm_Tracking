import torch
import numpy as np
import matplotlib.pyplot as plt
from pinnstorch.models import NetHFM, PINNModule
import pinnstorch
import sys

# --- Fix checkpoint dependency ---
def pde_fn(*args, **kwargs): return None
def output_fn(*args, **kwargs): return None
def read_data_fn(*args, **kwargs): return None

sys.modules['__main__'].pde_fn = pde_fn
sys.modules['__main__'].output_fn = output_fn
sys.modules['__main__'].read_data_fn = read_data_fn


def generate_comparison_plot(checkpoint_path, data_path):
    # =========================
    # 1. LOAD DATA (REFERENCE)
    # =========================
    data = pinnstorch.utils.load_data(data_path, "Aneurysm3D.mat")

    x_star = data["x_star"]
    y_star = data["y_star"]
    z_star = data["z_star"]
    t_star = data["t_star"]

    C_star = data["C_star"]
    U_star = data["U_star"]
    V_star = data["V_star"]
    P_star = data["P_star"]

    print("Data loaded successfully")

    # =========================
    # 2. PICK SAME TIME INDEX
    # =========================
    t_idx = 100   # MUST match config

    x = x_star[:, 0]
    y = y_star[:, 0]
    z = z_star[:, 0]

    # mid-plane slice
    z_mid = np.mean(z)
    mask = np.isclose(z, z_mid, atol=0.5)

    x_slice = x[mask]
    y_slice = y[mask]

    # =========================
    # 3. REFERENCE VALUES
    # =========================
    c_ref = C_star[mask, t_idx]
    u_ref = U_star[mask, t_idx]
    v_ref = V_star[mask, t_idx]
    p_ref = P_star[mask, t_idx]

    # =========================
    # 4. LOAD MODEL
    # =========================
    dummy_mean = np.zeros((1, 4), dtype=np.float32)
    dummy_std = np.ones((1, 4), dtype=np.float32)

    net = NetHFM(
        layers=[4, 128, 128, 128, 128, 5],
        output_names=["c", "u", "v", "w", "p"],
        mean=dummy_mean,
        std=dummy_std
    )

    model = PINNModule.load_from_checkpoint(
        checkpoint_path,
        net=net,
        map_location='cpu',
        weights_only=False
    )

    model.eval()
    print("Model loaded")

    # =========================
    # 5. MODEL PREDICTION
    # =========================
    x_t = torch.tensor(x_slice, dtype=torch.float32).view(-1, 1)
    y_t = torch.tensor(y_slice, dtype=torch.float32).view(-1, 1)
    z_t = torch.tensor(z[mask], dtype=torch.float32).view(-1, 1)
    t_value = float(t_star[0])  # force Python float (float32 compatible)
    t_t = torch.ones_like(x_t) * t_value

    with torch.no_grad():
        preds = model.net([x_t, y_t, z_t], t_t)

    c_pred = preds["c"].numpy().flatten()
    u_pred = preds["u"].numpy().flatten()
    v_pred = preds["v"].numpy().flatten()
    p_pred = preds["p"].numpy().flatten()

    # =========================
    # 6. INTERPOLATE TO GRID
    # =========================
    from scipy.interpolate import griddata

    xi = np.linspace(x_slice.min(), x_slice.max(), 200)
    yi = np.linspace(y_slice.min(), y_slice.max(), 200)

    grid_x, grid_y = np.meshgrid(xi, yi)

    def interp(values):
        return griddata(
            (x_slice, y_slice),
            values,
            (grid_x, grid_y),
            method='linear'   # IMPORTANT: not cubic
        )

    c_ref_g = interp(c_ref)
    c_pred_g = interp(c_pred)

    p_ref_g = interp(p_ref)
    p_pred_g = interp(p_pred)

    u_ref_g = interp(u_ref)
    v_ref_g = interp(v_ref)

    u_pred_g = interp(u_pred)
    v_pred_g = interp(v_pred)

    # =========================
    # 7. PLOTTING (FINAL OUTPUT)
    # =========================
    fig, ax = plt.subplots(3, 2, figsize=(12, 12))

    # Row 1: Concentration
    im0 = ax[0,0].contourf(grid_x, grid_y, c_ref_g, 50, cmap='magma')
    ax[0,0].set_title("Reference c")

    im1 = ax[0,1].contourf(grid_x, grid_y, c_pred_g, 50, cmap='magma')
    ax[0,1].set_title("Regressed c")

    # Row 2: Pressure
    im2 = ax[1,0].contourf(grid_x, grid_y, p_ref_g, 50, cmap='RdBu_r')
    ax[1,0].set_title("Reference p")

    im3 = ax[1,1].contourf(grid_x, grid_y, p_pred_g, 50, cmap='RdBu_r')
    ax[1,1].set_title("Regressed p")

    # Row 3: Streamlines
    ax[2,0].streamplot(grid_x, grid_y, u_ref_g, v_ref_g, color='k')
    ax[2,0].set_title("Reference streamlines")

    ax[2,1].streamplot(grid_x, grid_y, u_pred_g, v_pred_g, color='k')
    ax[2,1].set_title("Regressed streamlines")

    plt.tight_layout()
    plt.savefig("final_comparison.png")
    plt.show()

    print("✅ Final comparison plot saved!")


if __name__ == "__main__":
    checkpoint = r"C:\Users\abhig\OneDrive\Desktop\PINNs_Torch\examples\aneurysm3D\outputs\23-55-29\checkpoints\last.ckpt"
    data_path = r"C:\Users\abhig\OneDrive\Desktop\PINNs_Torch\data"

    generate_comparison_plot(checkpoint, data_path)