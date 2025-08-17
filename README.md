# GrocyGenie - AI Model

**Predict your groceries’ finishing dates with magic!**  
GrocyGenieModel is an AI-powered model designed to help you manage your kitchen efficiently by predicting when your grocery items are likely to run out. Say goodbye to unexpected shortages and last-minute grocery runs!  

This project is mainly built as a **side project** for our mobile app **"GrocieGenie"**.  
It leverages **Google Colab** for development and **Hugging Face** for model hosting and deployment.  


---

## Features

- **Predict Grocery Depletion:** Estimates the finishing date of each grocery item based on usage patterns.
- **Personalized Tracking:** Learns from your household consumption habits.
- **Supports Multiple Categories:** Works for perishables, dry goods, snacks, beverages, and more.
- **Easy Integration:** Can be integrated into apps, smart fridges, or personal assistant systems.  

---

## How It Works

1. **Data Collection:** Track usage frequency, purchase dates, quantities, and household size.  
2. **Model Training:** AI model learns consumption patterns using historical data.  
3. **Prediction:** Estimates the remaining days for each grocery item based on trends and patterns.  
4. **Notification:** Generates alerts or reports for items that will finish soon.  

The model uses advanced machine learning algorithms to capture patterns and variations in consumption, ensuring predictions are accurate and reliable.  

---

## Example Usage

```python
from grocygenie import GrocyGenieModel

# Initialize model
model = GrocyGenieModel()

# Input grocery data
grocery_data = [
    {"item": "Milk", "quantity": 2, "purchase_date": "2025-08-10"},
    {"item": "Rice", "quantity": 5, "purchase_date": "2025-07-25"},
    {"item": "Eggs", "quantity": 12, "purchase_date": "2025-08-12"},
]

# Predict finishing dates
predictions = model.predict_finishing_dates(grocery_data)

for item, finish_date in predictions.items():
    print(f"{item} will finish on: {finish_date}")

```
## Sample Output

```yaml
Milk will finish on: 2025-08-18
Rice will finish on: 2025-09-10
Eggs will finish on: 2025-08-20
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/GrocyGenieModel.git
cd GrocyGenieModel
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Run the script ( For Google Colab ):
```bash
python Model.py 
```

## Alternative Way by deploying on the HuggingFace
You can also deploy the model on Hugging Face and run it via API.
For this, we provide an example FastAPI app (app.py) where you can call the model through REST API requests.

## Model Training 
The model can be retrained with your personal grocery data:

```python
from grocygenie import GrocyGenieModel

model = GrocyGenieModel()
model.train(data_path="your_grocery_data.csv")
```
## Supported features for training:

- Item name

- Quantity purchased

- Purchase date

- Consumption frequency

- Household size (family mamber Count- Male ,Female, Children )

- Expiry dates ( Taken as feedback When needed )


Note: Development and experimentation were done primarily on Google Colab, and the model is hosted and maintained on Hugging Face for easy integration.

## Tech Stack

- Python 3.10.12 – Main programming language
- pandas & numpy – Data manipulation
- TensorFlow – Deep learning & model training
- Hugging Face – Model hosting and deployment
- Google Colab – Development and experimentation
- Flask/FastAPI (optional) – For API deployment

## Why GrocyGenieModel?

- Avoid waste by tracking consumption patterns.
- Save money by planning grocery purchases effectively.
- Keep your kitchen stocked intelligently without overbuying.
- Perfect for families, small restaurants, or personal smart kitchens.


## Future Improvements

- Mobile app integration with push notifications
- Predictive restocking recommendations
- Integration with online grocery shopping APIs
- Expiry date prediction for perishable items
- Smart analytics dashboard


## Contributions

Contributions, suggestions, and feature requests are welcome! Please open an issue or submit a pull request.

## Contact
- Email: **shahriarkabir280@gmail.com**




