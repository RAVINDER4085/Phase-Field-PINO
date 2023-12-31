import modulus
from modulus.hydra import instantiate_arch, ModulusConfig
from modulus.key import Key
from modulus.solver import Solver
from modulus.domain import Domain
from modulus.domain.constraint import SupervisedGridConstraint
from modulus.domain.validator import GridValidator
from modulus.dataset import DictGridDataset
from modulus.utils.io.plotter import GridValidatorPlotter
from utilities import load_FNO_dataset

@modulus.main(config_path="conf", config_name="fno")
def run(cfg: ModulusConfig) -> None:

    # load training/ test data
    input_keys =   input_keys = [
        Key("Ca"),
        Key("Cb"),
        ]
    output_keys = [Key("Microstructure")]

    invar_train, outvar_train = load_FNO_dataset(
        "dataset/data_train.hdf5",
        [k.name for k in input_keys],
        [k.name for k in output_keys],
        n_examples=1024,
    )
    invar_test, outvar_test = load_FNO_dataset(
        "dataset/data_test.hdf5",
        [k.name for k in input_keys],
        [k.name for k in output_keys],
        n_examples=256,
    )

    # make datasets
    train_dataset = DictGridDataset(invar_train, outvar_train)
    test_dataset = DictGridDataset(invar_test, outvar_test)

    # print out training/ test data shapes
    for d in (invar_train, outvar_train, invar_test, outvar_test):
        for k in d:
            print(f"{k}: {d[k].shape}")

    # make list of nodes to unroll graph on
    model = instantiate_arch(
        input_keys=input_keys,
        output_keys=output_keys,
        cfg=cfg.arch.fno,
    )
    nodes = model.make_nodes(name="FNO", jit=cfg.jit)

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
    )
    domain.add_validator(val, "test")

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()

if __name__ == "__main__":
    run()