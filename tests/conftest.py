import pytest

from grainsack.kg import KG

@pytest.fixture
def kg():
    return KG('Countries')

# @pytest.fixture
# def kges():
#     model = 