from src.smoke import main as smoke
def test_smoke() -> None:
    assert smoke() == True
