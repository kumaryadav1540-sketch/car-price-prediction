import pandas as pd
import pickle as pk
import streamlit as st

# -----------------------------
# Load trained model
# -----------------------------
with open("model.pkl", "rb") as file:
    model = pk.load(file)


# -----------------------------
# Load car dataset
# -----------------------------
cars_data = pd.read_csv("Cardetails.xls")


# -----------------------------
# Get car brand
# -----------------------------
def get_brand_name(car_name):
    return car_name.split(" ")[0].strip()


cars_data["name"] = cars_data["name"].apply(get_brand_name)


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🚗 Car Price Prediction ML Model")

name = st.selectbox(
    "Select Car Brand",
    sorted(cars_data["name"].unique())
)

year = st.slider(
    "Car Manufactured Year",
    min_value=1994,
    max_value=2024,
    value=2015
)

km_driven = st.number_input(
    "No. of Kilometers Driven",
    min_value=0,
    max_value=200000,
    value=50000
)

fuel = st.selectbox(
    "Fuel Type",
    cars_data["fuel"].unique()
)

seller_type = st.selectbox(
    "Seller Type",
    cars_data["seller_type"].unique()
)

transmission = st.selectbox(
    "Transmission Type",
    cars_data["transmission"].unique()
)

owner = st.selectbox(
    "Owner",
    cars_data["owner"].unique()
)

mileage = st.number_input(
    "Car Mileage",
    min_value=10.0,
    max_value=40.0,
    value=20.0
)

engine = st.number_input(
    "Engine CC",
    min_value=700,
    max_value=5000,
    value=1200
)

max_power = st.number_input(
    "Max Power",
    min_value=0.0,
    max_value=200.0,
    value=80.0
)

seats = st.number_input(
    "Number of Seats",
    min_value=2,
    max_value=10,
    value=5
)


# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Price"):

    # Convert categorical values to numbers
    owner_dict = {
        "First Owner": 1,
        "Second Owner": 2,
        "Third Owner": 3,
        "Fourth & Above Owner": 4,
        "Test Drive Car": 5
    }

    fuel_dict = {
        "Diesel": 1,
        "Petrol": 2,
        "LPG": 3,
        "CNG": 4
    }

    seller_dict = {
        "Individual": 1,
        "Dealer": 2,
        "Trustmark Dealer": 3
    }

    transmission_dict = {
        "Manual": 1,
        "Automatic": 2
    }

    brand_dict = {
        "Maruti": 1,
        "Skoda": 2,
        "Honda": 3,
        "Hyundai": 4,
        "Toyota": 5,
        "Ford": 6,
        "Renault": 7,
        "Mahindra": 8,
        "Tata": 9,
        "Chevrolet": 10,
        "Datsun": 11,
        "Jeep": 12,
        "Mercedes-Benz": 13,
        "Mitsubishi": 14,
        "Audi": 15,
        "Volkswagen": 16,
        "BMW": 17,
        "Nissan": 18,
        "Lexus": 19,
        "Jaguar": 20,
        "Land": 21,
        "MG": 22,
        "Volvo": 23,
        "Daewoo": 24,
        "Kia": 25,
        "Fiat": 26,
        "Force": 27,
        "Ambassador": 28,
        "Ashok": 29,
        "Isuzu": 30,
        "Opel": 31
    }

    # Convert selected values
    name_encoded = brand_dict.get(name)
    fuel_encoded = fuel_dict.get(fuel)
    seller_encoded = seller_dict.get(seller_type)
    transmission_encoded = transmission_dict.get(transmission)
    owner_encoded = owner_dict.get(owner)

    # Check if any value was not found
    if None in [
        name_encoded,
        fuel_encoded,
        seller_encoded,
        transmission_encoded,
        owner_encoded
    ]:
        st.error("Some input value is not available in the encoding dictionary.")
        st.stop()

    # Create input DataFrame
    input_data_model = pd.DataFrame(
        [[
            name_encoded,
            year,
            km_driven,
            fuel_encoded,
            seller_encoded,
            transmission_encoded,
            owner_encoded,
            mileage,
            engine,
            max_power,
            seats
        ]],
        columns=[
            "name",
            "year",
            "km_driven",
            "fuel",
            "seller_type",
            "transmission",
            "owner",
            "mileage",
            "engine",
            "max_power",
            "seats"
        ]
    )

    # Ensure everything is numeric
    input_data_model = input_data_model.astype(float)

    # Display input for debugging
    st.write("Input given to model:")
    st.write(input_data_model)

    # Predict
    try:
        car_price = model.predict(input_data_model)

        st.success(
            f"Estimated Car Price: ₹{car_price[0]:,.2f}"
        )

    except Exception as e:
        st.error(f"Prediction Error: {e}")