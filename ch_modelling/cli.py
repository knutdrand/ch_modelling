"""Console script for ch_modelling."""
from climate_health.datatypes import FullData
from cyclopts import App
from climate_health.data import DataSet

from ch_modelling.model import CHAPEstimator

app = App()

@app.command()
def train(training_data_filename: str, model_filename: str):
    '''
    '''
    dataset = DataSet.from_csv(training_data_filename, FullData)
    predictor = CHAPEstimator().train(dataset)
    predictor.save(model_filename)

@app.command()
def predict(model_filename: str, historic_data_filename: str, future_data_filename: str, output_filename: str):
    '''
    This function should just be type hinted with common types,
    and it will run as a command line function
    Simple function
    '''
    dataset = DataSet.from_csv(historic_data_filename)
    future_data = DataSet.from_csv(future_data_filename)

def main():
    typer.run(main_function)


if __name__ == "__main__":
    main()
