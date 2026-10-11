from dqn_carracing.cli import crear_parser


def test_cli_configura_el_subcomando_train():
    args = crear_parser().parse_args(["train", "--pasos", "123", "--sin-doble"])

    assert args.comando == "train"
    assert args.pasos == 123
    assert args.sin_doble is True
    assert callable(args.ejecutar)


def test_cli_configura_el_subcomando_evaluate():
    args = crear_parser().parse_args(["evaluate", "--aleatorio", "--episodios", "7"])

    assert args.comando == "evaluate"
    assert args.aleatorio is True
    assert args.episodios == 7
    assert callable(args.ejecutar)
