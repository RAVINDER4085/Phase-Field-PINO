from typing import Dict
import numpy as np
import torch
import torch.nn.functional as F
import modulus
from modulus.hydra import instantiate_arch, ModulusConfig
from modulus.key import Key
from modulus.models.layers import fourier_derivatives
from modulus.node import Node
from modulus.solver import Solver
from modulus.domain import Domain
from modulus.domain.constraint import SupervisedGridConstraint
from modulus.domain.validator import GridValidator
from modulus.dataset import DictGridDataset
from modulus.utils.io.plotter import GridValidatorPlotter
from modulus.utils.io.vtk import grid_to_vtk
from utilities import load_FNO_dataset
from ops import dx, ddx

class Cahn(torch.nn.Module):
    "Custom Cahn PDE definition for PINO"
    # we are only using gradient method as fourier
    # other two methods need few changes in data and cahn equation
    # as computing 4 order takes more computational power and memory

    def __init__(self, gradient_method: str = "hybrid"):
        super().__init__()
        self.gradient_method = str(gradient_method)

    def forward(self, input_var: Dict[str, torch.Tensor]) 
    -> Dict[str, torch.Tensor]:
        # get inputs
        u = input_var["Microstructure"]
        m = input_var["Ca"]
        c = input_var["Cb"]
        ga = input_var["gA_tilda"]
        gb = input_var["gB_tilda"]

        dxf = 1.0 / u.shape[-2]
        dyf = 1.0 / u.shape[-1]
        # Compute gradients based on method
        if self.gradient_method == "hybrid":
            dudx_exact = input_var["M__x"]
            dudy_exact = input_var["M__y"]
            dduddx_fdm = ddx(
                u, dx=dxf, channel=0, dim=0, order=1, 
                padding="replication")
            dduddy_fdm = ddx(
                u, dx=dyf, channel=0, dim=1, order=1, 
                padding="replication")
            # compute cahn equation
            cahn = (
                1.0
                - (ddgaddx_fdm) + (ddgaddy_fdm)
                + 8*(ddddmddddx_fdm + ddddmddddy_fdm )
                + 4*(ddddcddddx_fdm + ddddcddddy_fdm )
                - (ddgbddx_fdm) + (ddgbddy_fdm)
                + 8*(ddddcddddx_fdm + ddddcddddy_fdm )
                + 4*(ddddmddddx_fdm + ddddmddddy_fdm )
            )
        # FDM gradients
        elif self.gradient_method == "fdm":
            dcdx_fdm = dx(c, dx=dxf, channel=0, dim=0, order=1, 
            padding="replication")
            dcdy_fdm = dx(c, dx=dyf, channel=0, dim=1, order=1, 
            padding="replication")
            ddcddx_fdm = ddx(c, dx=dxf, channel=0, dim=0, order=1, 
            padding="replication")
            ddcddy_fdm = ddx(c, dx=dyf, channel=0, dim=1, order=1, 
            padding="replication")
            dcdxxxx_fdm = dddx(c, dx=dxf, channel=0, dim=0, order=1, 
            padding="replication")
            dcdyyyy_fdm = dddx(c, dx=dyf, channel=0, dim=1, order=1, 
            padding="replication")
            ddcddddx_fdm = ddddx(c, dx=dxf, channel=0, dim=0, order=1, 
            padding="replication")
            ddcddddy_fdm = ddddx(c, dx=dyf, channel=0, dim=1, order=1, 
            padding="replication")
            
            dmdxxxx_fdm = dddx(m, dx=dxf, channel=0, dim=0, order=1, 
            padding="replication")
            dmdyyyy_fdm = dddx(m, dx=dyf, channel=0, dim=1, order=1, 
            padding="replication")
            ddmddddx_fdm = ddddx(m, dx=dxf, channel=0, dim=0, order=1, 
            padding="replication")
            ddmddddy_fdm = ddddx(m, dx=dyf, channel=0, dim=1, order=1, 
            padding="replication")
            
            dgadx_fdm = dx(ga, dx=dxf, channel=0, dim=0, order=1, 
            padding="replication")
            dgady_fdm = dx(ga, dx=dyf, channel=0, dim=1, order=1, 
            padding="replication")
            ddgaddx_fdm = ddx(ga, dx=dxf, channel=0, dim=0, order=1, 
            padding="replication")
            ddgaddy_fdm = ddx(ga, dx=dyf, channel=0, dim=1, order=1, 
            padding="replication")
            
            dgbdx_fdm = dx(gb, dx=dxf, channel=0, dim=0, order=1, 
            padding="replication")
            dgbdy_fdm = dx(gb, dx=dyf, channel=0, dim=1, order=1, 
            padding="replication")
            ddgbddx_fdm = ddx(gb, dx=dxf, channel=0, dim=0, order=1, 
            padding="replication")
            ddgbddy_fdm = ddx(gb, dx=dyf, channel=0, dim=1, order=1, 
            padding="replication")
            
            # compute cahn equation
            cahn = (
                  1.0
                - (ddgaddx_fdm) + (ddgaddy_fdm)
                + 8*(ddddmddddx_fdm + ddddmddddy_fdm )
                + 4*(ddddcddddx_fdm + ddddcddddy_fdm )
                - (ddgbddx_fdm) + (ddgbddy_fdm)
                + 8*(ddddcddddx_fdm + ddddcddddy_fdm )
                + 4*(ddddmddddx_fdm + ddddmddddy_fdm )
            )
        # Fourier derivative
        # for this PINO we are only using gradient method fourier
        elif self.gradient_method == "fourier":
            dim_m_x = m.shape[2]
            dim_m_y = m.shape[3]
            dim_u_x = u.shape[2]
            dim_u_y = u.shape[3]
            dim_c_x = c.shape[2]
            dim_c_y = c.shape[3]
            dim_ga_x = ga.shape[2]
            dim_ga_y = ga.shape[3]
            dim_gb_x = gb.shape[2]
            dim_gb_y = gb.shape[3]
            m = F.pad(
                m, (0, dim_m_y - 1, 0, dim_m_x - 1), mode="reflect"
            ) 
            u = F.pad(
                u, (0, dim_u_y - 1, 0, dim_u_x - 1), mode="reflect"
            )  
            c = F.pad(
                c, (0, dim_c_y - 1, 0, dim_c_x - 1), mode="reflect"
            )  
            ga = F.pad(
                ga, (0, dim_ga_y - 1, 0, dim_ga_x - 1), mode="reflect"
            ) 
            gb = F.pad(
                gb, (0, dim_gb_y - 1, 0, dim_gb_x - 1), mode="reflect"
            ) 
            
            f_du, f_ddu = fourier_derivatives(u, [2.0, 2.0])
            f_du,f_ddddu = fourier_derivatives(u,[4.0, 4.0])
            f_dm, f_ddm = fourier_derivatives(m, [2.0, 2.0])
            f_dm,f_ddddm = fourier_derivatives(m,[4.0, 4.0])
            f_dc, f_ddddc = fourier_derivatives(c, [4.0, 4.0])
            f_dga, f_ddga = fourier_derivatives(ga, [2.0, 2.0])
            f_dgb, f_ddgb = fourier_derivatives(gb, [2.0, 2.0])
            dudx_fourier = f_du[:, 0:1, :dim_u_x, :dim_u_y]
            dudy_fourier = f_du[:, 1:2, :dim_u_x, :dim_u_y]
            dmdx_fourier = f_dm[:, 0:1, :dim_m_x, :dim_m_y]
            dmdy_fourier = f_dm[:, 1:2, :dim_m_x, :dim_m_y]
            ddgaddx_fourier = f_ddga[:, 0:1, :dim_u_x, :dim_u_y]
            ddgaddy_fourier = f_ddga[:, 1:2, :dim_u_x, :dim_u_y]
            ddgbddx_fourier = f_ddgb[:, 0:1, :dim_u_x, :dim_u_y]
            ddgbddy_fourier = f_ddgb[:, 1:2, :dim_u_x, :dim_u_y]
            dddduddddx_fourier = f_ddddu[:, 0:1, :dim_u_x, :dim_u_y]
            dddduddddy_fourier = f_ddddu[:, 1:2, :dim_u_x, :dim_u_y]
            ddddmddddx_fourier = f_ddddm[:, 0:1, :dim_m_x, :dim_m_y]
            ddddmddddy_fourier = f_ddddm[:, 1:2, :dim_m_x, :dim_m_y]
            ddddcddddx_fourier = f_ddddc[:, 0:1, :dim_c_x, :dim_c_y]
            ddddcddddy_fourier = f_ddddc[:, 1:2, :dim_c_x, :dim_c_y]
            # compute cahn equation
            cahn = (
                  1.0
                - (ddgaddx_fourier) + (ddgaddy_fourier)
                + 8*(ddddmddddx_fourier + ddddmddddy_fourier )
                + 4*(ddddcddddx_fourier + ddddcddddy_fourier )
                - (ddgbddx_fourier) + (ddgbddy_fourier)
                + 8*(ddddcddddx_fourier + ddddcddddy_fourier )
                + 4*(ddddmddddx_fourier + ddddmddddy_fourier )
                
                - (ddgbddx_fourier) + (ddgbddy_fourier)
                + 8*(ddddcddddx_fourier + ddddcddddy_fourier )
                + 4*(ddddmddddx_fourier + ddddmddddy_fourier )
                - (ddgaddx_fourier) + (ddgaddy_fourier)
                + 8*(ddddmddddx_fourier + ddddmddddy_fourier )
                + 4*(ddddcddddx_fourier + ddddcddddy_fourier )
                 
            )
        else:
            raise ValueError(f"Derivative method {self.gradient_method} 
            not supported.")

        # Zero outer boundary
        cahn = F.pad(cahn[:, :, 2:-2, 2:-2], [2, 2, 2, 2], "constant", 0)
        # Return cahn
        output_var = {
            "cahn": dxf * cahn,
        }  # weight boundary loss higher
        return output_var

# [pde-loss]

@modulus.main(config_path="conf", config_name="pino")
def run(cfg: ModulusConfig) -> None:

    # [datasets]
    # load training/ test data
    input_keys = [
        Key("Ca"),
        Key("Cb"),
        Key("gA_tilda"),
        Key("gB_tilda"),
    ]
    output_keys = [
        Key("Microstructure"),
    ]

    invar_train, outvar_train = load_FNO_dataset(
        "dataset/train.hdf5",
        [k.name for k in input_keys],
        [k.name for k in output_keys],
        n_examples=cfg.custom.ntrain,
    )
    invar_test, outvar_test = load_FNO_dataset(
        "dataset/test.hdf5",
        [k.name for k in input_keys],
        [k.name for k in output_keys],
        n_examples=cfg.custom.ntest,
    )

    # add additional constraining values for cahn variable
    outvar_train["cahn"] = np.zeros_like(outvar_train["Microstructure"])

    train_dataset = DictGridDataset(invar_train, outvar_train)
    test_dataset = DictGridDataset(invar_test, outvar_test)
    # [datasets]

    # [init-model]
    # Define FNO model
    if cfg.custom.gradient_method == "fourier":
        output_keys += [
            Key("Microstructure", derivatives=[Key("x")]),
            Key("Microstructure", derivatives=[Key("y")]),
        ]
    model = instantiate_arch(
        input_keys=[input_keys[0]],
        output_keys=output_keys,
        cfg=cfg.arch.fno,
        domain_length=[1.0, 1.0],
    )


    # Make custom cahn residual node for PINO
    inputs = [
        "Cb",
        "Ca",
        "gA_tilda",
        "gB_tilda",
        "Microstructure"
    ]
    if cfg.custom.gradient_method == "fourier":
        inputs += [
            "Microstructure__x",
            "Microstructure__y",
        ]
    cahn_node = Node(
        inputs=inputs,
        outputs=["cahn"],
        evaluate=Cahn(gradient_method=cfg.custom.gradient_method),
        name="Cahn Node",
    )
    nodes = model.make_nodes(name="FNO", jit=False) + [cahn_node]

    # make domain
    domain = Domain()

    # add constraints to domain
    supervised = SupervisedGridConstraint(
        nodes=nodes,
        dataset=train_dataset,
        batch_size=cfg.batch_size.grid,
    )
    domain.add_constraint(supervised, "supervised")

    # add validator
    val = GridValidator(
        nodes,
        dataset=test_dataset,
        batch_size=cfg.batch_size.validation,
        plotter=GridValidatorPlotter(n_examples=5),
        requires_grad=True,
    )
    domain.add_validator(val, "test")

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()