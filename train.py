from typing import Any, Dict, List, Optional, Tuple
import hydra
import numpy as np
import torch
import pinnstorch
from omegaconf import DictConfig


def read_data_fn(root_path):
    """
    Loads Aneurysm3D data.
    We MUST include 'c' here because your config.yaml expects it.
    """
    data = pinnstorch.utils.load_data(root_path, "Aneurysm3D.mat")

    return pinnstorch.data.PointCloudData(
        spatial=[data["x_star"], data["y_star"], data["z_star"]],
        time=[data["t_star"]],
        solution={
            "u": data["U_star"], 
            "v": data["V_star"], 
            "w": data["W_star"], 
            "p": data["P_star"], 
            "c": data["C_star"]  # Keep this as 'c' to fix the KeyError
        },
    )
def output_fn(outputs: Dict[str, torch.Tensor], x, y, z, t):
    # Create a COPY for the scan so we don't break the 3D 'c' size
    outputs["c_scanned"] = torch.mean(outputs["c"], dim=0, keepdim=True)
    return outputs

def pde_fn(outputs: Dict[str, torch.Tensor], x, y, z, t):   
    Pec = 1.0 / 0.0101822
    Rey = 1.0 / 0.0101822

    # Now outputs["c"] is still size 10,000, so this cat will work!
    Y = torch.cat([outputs["c"], outputs["u"], outputs["v"], outputs["w"], outputs["p"]], 1)
    
    # ... rest of your PDE code (Y_t, Y_x, residuals e1-e5) ...
    # Ensure you keep the split and equations exactly as they were in your original code
    
    Y_t, Y_x, Y_y, Y_z = pinnstorch.utils.fwd_gradient(Y, [t, x, y, z])
    Y_xx = pinnstorch.utils.fwd_gradient(Y_x, x)[0]
    Y_yy = pinnstorch.utils.fwd_gradient(Y_y, y)[0]
    Y_zz = pinnstorch.utils.fwd_gradient(Y_z, z)[0]

    c, u, v, w, p = torch.split(Y, (1), dim=1)
    c_t, u_t, v_t, w_t, _ = torch.split(Y_t, (1), dim=1)
    c_x, u_x, v_x, w_x, p_x = torch.split(Y_x, (1), dim=1)
    c_y, u_y, v_y, w_y, p_y = torch.split(Y_y, (1), dim=1)
    c_z, u_z, v_z, w_z, p_z = torch.split(Y_z, (1), dim=1)
    c_xx, u_xx, v_xx, w_xx, _ = torch.split(Y_xx, (1), dim=1)
    c_yy, u_yy, v_yy, w_yy, _ = torch.split(Y_yy, (1), dim=1)
    c_zz, u_zz, v_zz, w_zz, _ = torch.split(Y_zz, (1), dim=1)

    outputs["e1"] = c_t + (u * c_x + v * c_y + w * c_z) - (1.0 / Pec) * (c_xx + c_yy + c_zz)
    outputs["e2"] = u_t + (u * u_x + v * u_y + w * u_z) + p_x - (1.0 / Rey) * (u_xx + u_yy + u_zz)
    outputs["e3"] = v_t + (u * v_x + v * v_y + w * v_z) + p_y - (1.0 / Rey) * (v_xx + v_yy + v_zz)
    outputs["e4"] = w_t + (u * w_x + v * w_y + w * w_z) + p_z - (1.0 / Rey) * (w_xx + w_yy + w_zz)
    outputs["e5"] = u_x + v_y + w_z

    return outputs
@hydra.main(version_base="1.3", config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    pinnstorch.utils.extras(cfg)

    # Capture the output_dict
    metric_dict, object_dict = pinnstorch.train(
        cfg, read_data_fn=read_data_fn, pde_fn=pde_fn, output_fn=output_fn
    )

    # FORCE SAVE LINE: This creates the file you need for the graph script
    trainer = object_dict["trainer"]
    trainer.save_checkpoint("force_final.ckpt")
    print("--- CHECKPOINT SAVED AS force_final.ckpt ---")

    metric_value = pinnstorch.utils.get_metric_value(
        metric_dict=metric_dict, metric_names=cfg.get("optimized_metric")
    )
    return metric_value

if __name__ == "__main__":
    main()