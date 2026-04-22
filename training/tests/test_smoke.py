def test_package_imports() -> None:
    import progno_train
    import progno_train.cli

    assert progno_train is not None
    assert progno_train.cli.main is not None
