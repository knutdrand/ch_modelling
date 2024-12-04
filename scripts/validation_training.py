from chap_core.assessment.dataset_splitting import train_test_generator
from chap_core.data.datasets import ISIMIP_dengue_harmonized

from ch_modelling.tuned_models import ar_model_monthly_v1, ar_model_monthly_v2


def monthly_validated_train(dataset, model, level=2):
    train, test = train_test_generator(dataset, prediction_length=12, n_test_sets=1)
    if level == 2:
        train, validation = train_test_generator(train, prediction_length=12, n_test_sets=1)
    else:
        validation = test

    t = next(validation)
    print(t)
    historic_data, _, future_data = t
    model.set_validation_data(historic_data, future_data)
    model.train(train)


if __name__ == "__main__":
    country_name = 'mexico'
    model = ar_model_monthly_v2()
    dataset = dataset = ISIMIP_dengue_harmonized[country_name]
    monthly_validated_train(dataset, model, level=2)
