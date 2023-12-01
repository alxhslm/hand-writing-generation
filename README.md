# :writing_hand: Hand writing generation
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hand-writing-generation.streamlit.app/)

The objective of this project is to be able to:
1. Classify hand-written characters
2. Generate realistic hand-written characters in a given style

## Downloading the dataset
The dataset can be download from [Kaggle](https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format/data). You can either download this:
- Directly from the website
- Using the Kaggle API as follows:
    ``` bash
    !kaggle datasets download -d sachinpatel21/az-handwritten-alphabets-in-csv-format -p data --unzip
    ```

## Using the interactive dashboard
To classify your own hand-written characters and generate synthetic characters in your own hand-writing style, you can use the interactive `streamlit` dashboard. This is hosted on [Streamlit Cloud](https://streamlit.io/cloud).

You can also launch the dashboard locally by running the following command:

``` bash
streamlit run dashboard.py
```
