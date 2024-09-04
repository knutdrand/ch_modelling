"""Console script for ch_modelling."""
from cyclopts import App
from climate_health.data import DataSet

app = App()

@app.command()
def train(training_data_filename: str):
    '''
    This function should just be type hinted with common types,
    and it will run as a command line function
    Simple function
    '''
    dataset = DataSet.from_csv(training_data_filename)


def main():
    typer.run(main_function)


if __name__ == "__main__":
    main()
