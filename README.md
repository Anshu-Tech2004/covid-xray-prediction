# COVID-19 X-ray Prediction

This project uses a deep learning model (ResNet18) to predict COVID-19 from chest X-ray images.

## Installation

1.  Clone the repository:

    ```bash
    git clone [https://github.com/Anshu-Tech2004/covid-xray-prediction.git](https://github.com/Anshu-Tech2004/covid-xray-prediction.git)
    ```

2.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3.  Download the dataset: [https://storage.googleapis.com/kaggle-data-sets/818155/1400262/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250331%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250331T080545Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=1d0a67dba4dd9dab8978596ee76b2a92fdb326388c280cf3fb1b6745b4623df5e28630817dd5e7c39320a4748dc503c08ca20b24c99a40ae51047a3d870eceb85878df4688d590ac97cd6634efc206bd062307b1192c3c7581abe6c3d658bc7addfa79d0dcc29be543c5d89b6640f6da6b70d3c197daa89f9d1a3990717bb4f2192d2a5dc369ceec51e6598dcdb1bfeb2093bef297e370ccbd4694bee0e803f28ecf453ece8979d5714f8a7b5ee75646922e754cc85a8290e3d4b5ccd29668fd669a16b6aa2de4084673a9245a9c2080540eb2ec281c3dbbfcf42a36e38b157d6f64565cb0b8be9f8685724c206379aec018ee58a78f2019560ee95613e03969]](https://storage.googleapis.com/kaggle-data-sets/818155/1400262/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250331%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250331T080545Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=1d0a67dba4dd9dab8978596ee76b2a92fdb326388c280cf3fb1b6745b4623df5e28630817dd5e7c39320a4748dc503c08ca20b24c99a40ae51047a3d870eceb85878df4688d590ac97cd6634efc206bd062307b1192c3c7581abe6c3d658bc7addfa79d0dcc29be543c5d89b6640f6da6b70d3c197daa89f9d1a3990717bb4f2192d2a5dc369ceec51e6598dcdb1bfeb2093bef297e370ccbd4694bee0e803f28ecf453ece8979d5714f8a7b5ee75646922e754cc85a8290e3d4b5ccd29668fd669a16b6aa2de4084673a9245a9c2080540eb2ec281c3dbbfcf42a36e38b157d6f64565cb0b8be9f8685724c206379aec018ee58a78f2019560ee95613e03969)
    4.  Download the trained model: [covid_model.pth](covid_model.pth)

    ## Usage

    1.  Run the Streamlit app:

        ```bash
        streamlit run streamlit_app.py
        ```

    ## Notes

    * Make sure you have the dataset and model in the same directory as the script.
    * If you encounter any issues, make sure you have python version 3.7 or higher installed.
    * It is recommended to use a virtual environment.