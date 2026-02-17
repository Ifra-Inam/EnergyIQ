# EnergyIQ

**EnergyIQ** is a building energy efficiency prediction app. It takes building features as input and predicts the **heating and cooling loads**. The app provides a comparison chart against average, most efficient, and least efficient buildings, along with **AI insights** that include interpretation of the prediction, three actionable recommendations, and estimated potential savings.

---

## Features

- Input features:
  - Relative Compactness
  - Surface Area
  - Wall Area
  - Roof Area
  - Overall Height
  - Orientation
  - Glazing Area
  - Glazing Area Distribution
- Predicts heating and cooling loads
- Displays comparison chart: predicted vs average, most efficient, and least efficient
- AI insights including:
  - Interpretation of prediction
  - Three actionable recommendations
  - Estimated potential savings

---

## Dataset

Dataset: Tsanas, A. & Xifara, A. (2012). Energy Efficiency [Dataset]. UCI Machine Learning Repository. [DOI: 10.24432/C51307](https://doi.org/10.24432/C51307)

---

## Installation and Setup

1. Clone the repository:

```bash
git clone https://github.com/Ifra-Inam/EnergyIQ.git
cd EnergyIQ
```

2. Create a virtual environment:

```bash
python -m venv venv
```

3. Activate virtual environment:

```bash
venv\Scripts\activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Train the model:

```bash
python training.py
```

2. Create a .env file in the root directory with your GROQ API key:

GROQ_API_KEY=your_api_key_here

3. Run the Streamlit app:

```bash
streamlit run app.py
```

## How It Works

1. User inputs building features.
2. App predicts heating and cooling loads.
3. Displays a comparison chart against average, most efficient, and least efficient buildings.
4. Provides AI insights:
  - Interpretation of the prediction
  - Three actionable recommendations
  - Estimated potential savings

## License
This project is licensed under the MIT License — feel free to use, modify, and distribute it.
