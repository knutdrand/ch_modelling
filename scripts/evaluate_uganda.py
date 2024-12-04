from chap_core.assessment.prediction_evaluator import evaluate_model
from chap_core.data.datasets import ISIMIP_dengue_harmonized
from chap_core.external.external_model import get_model_from_directory_or_github_url

from ch_modelling.tuned_models import ar_model_monthly_v2, ar_model_monthly_v1

models = {
    #'chap_ewars': get_model_from_directory_or_github_url(
    #"https://github.com/sandvelab/chap_auto_ewars"),
    'ar_model_monthly_v1': ar_model_monthly_v1(),
    'ar_model_monthly_v2': ar_model_monthly_v2()}




country_name = "vietnam"
#model_name = 'ar_model_monthly_v1'
#model_name = 'chap_ewars'
for model_name in models:
    model = models[model_name]
    dataset = DataSet.from_csv('/home/knut/Data/', FullData)
    #model = ar_model_monthly_v1()
    model = models[model_name]
    results = evaluate_model(model, dataset, prediction_length=3, n_test_sets=10,
                             report_filename=f"{model_name}_{country_name}_monthly.pdf")
    print(results)

# Vietnam
# ({'MSE': 15742.770235438597, 'abs_error': 42386.0, 'abs_target_sum': 116543.0, 'abs_target_mean': 204.46140350877192, 'seasonal_error': 136.04230138455713, 'MASE': 0.8008088370317368, 'MAPE': 0.39412235668921275, 'sMAPE': 0.4487559879524475, 'MSIS': 12.856736667951408, 'num_masked_target_values': 0.0, 'QuantileLoss[0.1]': 19394.0, 'Coverage[0.1]': 0.0543859649122807, 'QuantileLoss[0.5]': 42386.0, 'Coverage[0.5]': 0.37192982456140344, 'QuantileLoss[0.9]': 35426.79999999999, 'Coverage[0.9]': 1.0, 'RMSE': 125.47019660237484, 'NRMSE': 0.6136620137061313, 'ND': 0.363694087160962, 'wQuantileLoss[0.1]': 0.16641068103618406, 'wQuantileLoss[0.5]': 0.363694087160962, 'wQuantileLoss[0.9]': 0.30398050504963825, 'mean_absolute_QuantileLoss': 32402.266666666663, 'mean_wQuantileLoss': 0.2780284244155948, 'MAE_Coverage': 0.5, 'OWA': nan},     item_id forecast_start  ...  QuantileLoss[0.9]  Coverage[0.9]
# Brazil
# ({'MSE': 2319852.417122174, 'abs_error': 401408.0, 'abs_target_sum': 587300.0, 'abs_target_mean': 851.1594202898551, 'seasonal_error': 2193.1098349541885, 'MASE': 0.39362840568969226, 'MAPE': 1.219974602350388, 'sMAPE': 0.6107707589312357, 'MSIS': 3.4996303916376243, 'num_masked_target_values': 0.0, 'QuantileLoss[0.1]': 95100.20000000003, 'Coverage[0.1]': 0.05072463768115942, 'QuantileLoss[0.5]': 401408.0, 'Coverage[0.5]': 0.6782608695652174, 'QuantileLoss[0.9]': 330915.79999999993, 'Coverage[0.9]': 0.9797101449275363, 'RMSE': 1523.106173949201, 'NRMSE': 1.7894487655796845, 'ND': 0.6834803337306317, 'wQuantileLoss[0.1]': 0.1619278052102844, 'wQuantileLoss[0.5]': 0.6834803337306317, 'wQuantileLoss[0.9]': 0.5634527498722969, 'mean_absolute_QuantileLoss': 275808.0, 'mean_wQuantileLoss': 0.469620296271071, 'MAE_Coverage': 0.4797101449275362, 'OWA': nan},     item_id forecast_start  ...  QuantileLoss[0.9]  Coverage[0.9]

# Vietnam
# ({'MSE': 41180.992733157895, 'abs_error': 47263.0, 'abs_target_sum': 116543.0, 'abs_target_mean': 204.46140350877192, 'seasonal_error': 136.04230138455713, 'MASE': 0.5904412196668878, 'MAPE': 0.46368929003299486, 'sMAPE': 0.6026516659163224, 'MSIS': 10.521393649954685, 'num_masked_target_values': 0.0, 'QuantileLoss[0.1]': 19787.0, 'Coverage[0.1]': 0.05263157894736842, 'QuantileLoss[0.5]': 47263.0, 'Coverage[0.5]': 0.2649122807017543, 'QuantileLoss[0.9]': 47160.2, 'Coverage[0.9]': 0.9385964912280701, 'RMSE': 202.93100485918333, 'NRMSE': 0.9925149753287156, 'ND': 0.4055413023519216, 'wQuantileLoss[0.1]': 0.16978282693941293, 'wQuantileLoss[0.5]': 0.4055413023519216, 'wQuantileLoss[0.9]': 0.40465922449224745, 'mean_absolute_QuantileLoss': 38070.066666666666, 'mean_wQuantileLoss': 0.3266611179278606, 'MAE_Coverage': 0.43859649122807015, 'OWA': nan},     item_id forecast_start  ...  QuantileLoss[0.9]  Coverage[0.9]
# ({'MSE': 1546892.8883901448, 'abs_error': 344657.0, 'abs_target_sum': 587300.0, 'abs_target_mean': 851.1594202898551, 'seasonal_error': 2193.1098349541885, 'MASE': 0.35509607142680283, 'MAPE': 0.9550788283203993, 'sMAPE': 0.6299591216008357, 'MSIS': 2.600882526770624, 'num_masked_target_values': 0.0, 'QuantileLoss[0.1]': 99340.20000000001, 'Coverage[0.1]': 0.09420289855072464, 'QuantileLoss[0.5]': 344657.0, 'Coverage[0.5]': 0.4695652173913043, 'QuantileLoss[0.9]': 233858.8, 'Coverage[0.9]': 0.8492753623188405, 'RMSE': 1243.7414877659041, 'NRMSE': 1.4612321242269264, 'ND': 0.5868499914864634, 'wQuantileLoss[0.1]': 0.16914728418184916, 'wQuantileLoss[0.5]': 0.5868499914864634, 'wQuantileLoss[0.9]': 0.39819308700834327, 'mean_absolute_QuantileLoss': 225952.0, 'mean_wQuantileLoss': 0.38473012089221864, 'MAE_Coverage': 0.38309178743961353, 'OWA': nan},     item_id forecast_start  ...  QuantileLoss[0.9]  Coverage[0.9]

# Thailand
# ({'MSE': 5818.694114642856, 'abs_error': 21612.0, 'abs_target_sum': 40752.0, 'abs_target_mean': 48.54166666666666, 'seasonal_error': 59.02386030859135, 'MASE': 0.3888250248684377, 'MAPE': 0.9600856552443188, 'sMAPE': 0.5823542127958306, 'MSIS': 3.4702557780818095, 'num_masked_target_values': 33.0, 'QuantileLoss[0.1]': 6574.0, 'Coverage[0.1]': 0.06607142857142857, 'QuantileLoss[0.5]': 21612.0, 'Coverage[0.5]': 0.5988095238095238, 'QuantileLoss[0.9]': 16825.6, 'Coverage[0.9]': 0.9488095238095239, 'RMSE': 76.2803651973616, 'NRMSE': 1.5714409997739733, 'ND': 0.5303297997644287, 'wQuantileLoss[0.1]': 0.16131723596387906, 'wQuantileLoss[0.5]': 0.5303297997644287, 'wQuantileLoss[0.9]': 0.41287789556340787, 'mean_absolute_QuantileLoss': 15003.866666666667, 'mean_wQuantileLoss': 0.3681749770972385, 'MAE_Coverage': 0.44880952380952394, 'OWA': nan},     item_id forecast_start  ...  QuantileLoss[0.9]  Coverage[0.9]
# ({'MSE': 10279.206912083335, 'abs_error': 28727.0, 'abs_target_sum': 40752.0, 'abs_target_mean': 48.54166666666666, 'seasonal_error': 59.02386030859135, 'MASE': 0.49410350495776206, 'MAPE': 0.8688913021023548, 'sMAPE': 0.6643245164027146, 'MSIS': 4.443955274229694, 'num_masked_target_values': 33.0, 'QuantileLoss[0.1]': 7341.200000000001, 'Coverage[0.1]': 0.07023809523809522, 'QuantileLoss[0.5]': 28727.0, 'Coverage[0.5]': 0.5119047619047619, 'QuantileLoss[0.9]': 25559.799999999996, 'Coverage[0.9]': 0.8404761904761904, 'RMSE': 101.38642370694083, 'NRMSE': 2.0886473553361204, 'ND': 0.7049224577934825, 'wQuantileLoss[0.1]': 0.18014330585001964, 'wQuantileLoss[0.5]': 0.7049224577934825, 'wQuantileLoss[0.9]': 0.6272035728307812, 'mean_absolute_QuantileLoss': 20542.666666666664, 'mean_wQuantileLoss': 0.5040897788247611, 'MAE_Coverage': 0.38015873015873014, 'OWA': nan},     item_id forecast_start  ...  QuantileLoss[0.9]  Coverage[0.9]
