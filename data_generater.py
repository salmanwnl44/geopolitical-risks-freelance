import pandas as pd
import random
from faker import Faker

# Initialize Faker for random data generation
fake = Faker()
Faker.seed(42)  # Ensure reproducibility

# Define possible categories
countries = ["USA", "China", "Russia", "Germany", "India", "UK", "Brazil", "France", "Japan", "South Africa"]
event_types = ["War", "Sanctions", "Elections", "Cyber Attack", "Terrorism", "Trade Dispute", "Pandemic"]
risk_categories = ["Political", "Economic", "Military", "Environmental", "Cybersecurity"]
sources = ["News", "Government Report", "Social Media", "Intelligence"]

# Generate 100,000 records
num_records = 100000
data = []

for _ in range(num_records):
    event_id = fake.uuid4()
    date = fake.date_between(start_date="-5y", end_date="today")  # Last 5 years
    country = random.choice(countries)
    event_type = random.choice(event_types)
    risk_category = random.choice(risk_categories)
    severity_score = random.randint(1, 100)  # Risk level (1-100)
    sentiment_score = round(random.uniform(-1, 1), 2)  # -1 (Negative) to +1 (Positive)
    revenue_loss = round(random.uniform(0, 10), 2)  # Revenue loss percentage
    supply_delay = round(random.uniform(0, 30), 2)  # Supply chain delay in days
    market_impact = round(random.uniform(-5, 5), 2)  # Stock market change in %
    source = random.choice(sources)

    data.append([event_id, date, country, event_type, risk_category, severity_score, sentiment_score,
                 revenue_loss, supply_delay, market_impact, source])

# Create DataFrame
df = pd.DataFrame(data, columns=["Event_ID", "Date", "Country", "Event_Type", "Risk_Category", 
                                 "Severity_Score", "Sentiment_Score", "Revenue_Loss%", "Supply_Delay_Days", 
                                 "Market_Impact%", "Source"])

# Save as CSV
df.to_csv("geopolitical_risk_data.csv", index=False)
print("Dataset Generated: 100,000 Records Saved as 'geopolitical_risk_data.csv'")
