import streamlit as st
import pandas as pd
import pickle
final_pl = pickle.load(open(r"estimator1.pkl", 'rb'))

# Streamlit page title
st.title("Travel Prediction App")

# Input fields for user data
num_passengers = st.number_input("Number of Passengers", min_value=1, max_value=10, value=2)
sales_channel = st.selectbox("Sales Channel", ["Internet", "Agent"])
trip_type = st.selectbox("Trip Type", ["RoundTrip", "OneWay"])
purchase_lead = st.number_input("Purchase Lead (days)", min_value=0, max_value=365, value=262)
length_of_stay = st.number_input("Length of Stay (days)", min_value=1, max_value=100, value=19)
flight_hour = st.number_input("Flight Hour", min_value=0, max_value=23, value=7)
flight_day = st.number_input("Flight Day", min_value=1, max_value=31, value=6)
route = st.text_input("Route", value="AKLDEL")
booking_origin = st.text_input("Booking Origin", value="New Zealand")
wants_extra_baggage = st.selectbox("Wants Extra Baggage", [0, 1])
wants_preferred_seat = st.selectbox("Wants Preferred Seat", [0, 1])
wants_in_flight_meals = st.selectbox("Wants In-flight Meals", [0, 1])
flight_duration = st.number_input("Flight Duration (hours)", min_value=0.0, max_value=24.0, value=5.52)
parts_of_the_day = st.selectbox("Part of the Day", ["Morning", "Afternoon", "Evening", "Night"])

# Handle Purchase Lead Bin with possible error due to unknown categories
purchase_lead_bin = st.selectbox(
    "Purchase Lead Bin",
    [ "15-30 days", "30+ days"]
)
all_services_opted = st.selectbox("All Services Opted", [0, 1])

# Organize input data
input_data = pd.DataFrame([[num_passengers, sales_channel, trip_type, purchase_lead, length_of_stay, 
                            flight_hour, flight_day, route, booking_origin, wants_extra_baggage, 
                            wants_preferred_seat, wants_in_flight_meals, flight_duration, 
                            parts_of_the_day, purchase_lead_bin, all_services_opted]],
                          columns=['num_passengers', 'sales_channel', 'trip_type', 'purchase_lead',
                                   'length_of_stay', 'flight_hour', 'flight_day', 'route',
                                   'booking_origin', 'wants_extra_baggage', 'wants_preferred_seat',
                                   'wants_in_flight_meals', 'flight_duration','parts_of_the_day', 
                                   'purchase_lead_bin', 'all_services_opted'])

# Display the prediction when the button is clicked
if st.button("Predict"):
    try:
        # Predict using the loaded pipeline model
        predicted_value = final_pl.predict(input_data)
        st.write("Predicted Value:", predicted_value[0])
    except ValueError as e:
        st.error("An error occurred during prediction. Please check input values.")
        st.write("Error details:", e)
