import pytest
from analysis.studies.four_rooms import common_steps, paired_probability, ARMS


def test_no_forward_fill_or_partial_seed_comparison():
    runs={(a,s):{0:{},10:{},20:{}} for a in ARMS for s in range(5)}
    del runs['cce',4][20]
    assert common_steps(runs)==[0,10]
    del runs['cce',4]
    assert common_steps(runs)==[]


def test_paired_probability():
    assert paired_probability([1]*5)==1
    assert paired_probability([-1]*5)==0
    assert paired_probability([0]*5)==0
    with pytest.raises(ValueError): paired_probability([1]*4)
