# COVID-19 X-ray Prediction

This project uses a deep learning model (ResNet18) to predict COVID-19 from chest X-ray images.

## Installation

1.  Clone the repository:

    ```bash
    git clone [https://github.com/Anshu-Tech2004/covid-xray-prediction.git](https://github.com/Anshu-Tech2004/covid-xray-prediction.git)
    ```

2.  **Create a virtual environment (Recommended):**

    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**

    * **Windows:**

        ```bash
        venv\Scripts\activate
        ```

    * **macOS/Linux:**

        ```bash
        source venv/bin/activate
        ```

4.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

5.  Download the dataset: [https://drive.google.com/drive/folders/1bHjgmQ867DrFveZbQ4BA80JsYVIEBEev?usp=sharing](https://drive.google.com/drive/folders/1bHjgmQ867DrFveZbQ4BA80JsYVIEBEev?usp=sharing)
6.  Download the trained model: [https://drive.google.com/file/d/1bpyudVuSskF5S8csG_iu2fHir042S9Qp/view?usp=sharing](https://drive.google.com/file/d/1bpyudVuSskF5S8csG_iu2fHir042S9Qp/view?usp=sharing)

## Usage

1.  Run the Streamlit app:

    ```bash
    streamlit run streamlit_app.py
    ```

2.  **Deactivate the virtual environment (when finished):**

    ```bash
    deactivate
    ```

## Notes

* Make sure you have the dataset and model in the same directory as the script.
* The trained model `covid_model.pth` needs to be downloaded from the Google Drive link provided above.
* If you encounter any issues, make sure you have Python version 3.7 or higher installed.
* It is highly recommended to use a virtual environment.