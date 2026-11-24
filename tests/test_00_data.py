from scripts.check_data import check


def test_data_files_agree():
    assert check() == []
