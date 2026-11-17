import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.datos import BASE, validate
from src.pipeline import PhysicalFeatures, make_pipeline
from src.metricas import choose_threshold

def frame():
    return pd.DataFrame({'product_type':['L','M','H','L'], 'air_temperature_k':[300.,301,302,303],
        'process_temperature_k':[310.,311,312,313],'rotational_speed_rpm':[1200.,1300,1400,1500],
        'torque_nm':[30.,40,50,60],'tool_wear_min':[10.,20,30,40]})

def test_leakage_columns_are_discarded_and_features_are_physical():
    x=frame();x['machine_failure']=[0,1,0,1];x['TWF']=[0,1,0,1];x['baseline_alert']=1;x['reading_id']=100
    result=PhysicalFeatures().transform(x)
    assert set(result)==set(BASE+['temperature_delta_k','power_w'])
    np.testing.assert_allclose(result.temperature_delta_k,10)
    np.testing.assert_allclose(result.power_w,x.torque_nm*x.rotational_speed_rpm*2*np.pi/60)

def test_imputer_is_fit_on_training_only():
    x=frame();x.loc[0,'torque_nm']=np.nan
    pipe=make_pipeline(LogisticRegression()).fit(x,np.array([0,1,0,1]))
    before=pipe.named_steps['prepare'].named_transformers_['numeric'].named_steps['imputer'].statistics_.copy()
    test=frame();test['torque_nm']=1e9;pipe.predict(test)
    after=pipe.named_steps['prepare'].named_transformers_['numeric'].named_steps['imputer'].statistics_
    np.testing.assert_equal(before,after);assert before[3]==50

def test_threshold_respects_capacity():
    y=np.array([0]*90+[1]*10);prob=np.linspace(0,1,100)
    threshold=choose_threshold(y,prob)
    assert (prob>=threshold).mean()<=.12

def test_obt_rejects_duplicate_grain():
    import pytest
    x=frame();x['reading_id']=[1,1,2,3];x['machine_failure']=[0,1,0,1]
    with pytest.raises(ValueError,match='Grano'):validate(x)
