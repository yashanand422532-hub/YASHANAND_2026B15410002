import pandas as pd 
import numpy as np

# Load the hotel booking data
df = pd.read_csv("hotel_bookings_1200.csv")

print("First five records")
print(df.head())

print("Dataset information")
print(df.info())

print("Total records:", len(df))
print("Total columns:", len(df.columns))

# Check missing values and duplicate booking IDs
print("Missing values")
print(df.isnull().sum())

print("Duplicate booking IDs:", df["Booking_ID"].duplicated().sum())

# Remove duplicate booking IDs
df = df.drop_duplicates(subset="Booking_ID") 

# Fill missing lead time with the median
df["Lead_Time_Days"] = df["Lead_Time_Days"].fillna(
    df["Lead_Time_Days"].median()
)

# Fill missing room rate with the median rate of the same room type
df["Room_Rate"] = df["Room_Rate"].fillna(
    df.groupby("Room_Type")["Room_Rate"].transform("median")
)

print("Missing values after cleaning")
print(df.isnull().sum())

# Change check-in date into date format
df["Checkin_Date"] = pd.to_datetime(df["Checkin_Date"])

# Get the month and weekday from the check-in date
df["Arrival_Month"] = df["Checkin_Date"].dt.month_name()
df["Arrival_Weekday"] = df["Checkin_Date"].dt.day_name()

# Calculate total bookings, room nights and cancellations
total_bookings = len(df)
total_room_nights = df["Nights"].sum()
canceled_bookings = (df["Canceled"] == "Yes").sum()

cancellation_percentage = (
    canceled_bookings / total_bookings
) * 100

print("Total bookings:", total_bookings)
print("Total room nights:", total_room_nights)
print("Canceled bookings:", canceled_bookings)
print("Cancellation percentage:", round(cancellation_percentage, 2), "%")

# Calculate revenue for each booking
df["Booking_Revenue"] = df["Nights"] * df["Room_Rate"]

# Keep only bookings that were not cancelled
completed = df[df["Canceled"] == "No"].copy()

total_revenue = completed["Booking_Revenue"].sum()
print("Total revenue: Rs.", round(total_revenue, 2))

# Compare room types
room_summary = completed.groupby("Room_Type").agg(
    Total_Bookings=("Booking_ID", "count"),
    Average_Room_Rate=("Room_Rate", "mean"),
    Total_Room_Nights=("Nights", "sum"),
    Total_Revenue=("Booking_Revenue", "sum")
).reset_index()

print("Room type summary")
print(room_summary.round(2))

# Compare booking channels
channel_summary = completed.groupby("Booking_Channel").agg(
    Total_Bookings=("Booking_ID", "count"),
    Total_Revenue=("Booking_Revenue", "sum")
).reset_index()

channel_summary = channel_summary.sort_values(
    by="Total_Bookings", ascending=False
)

print("Booking channel ranking")
print(channel_summary.round(2))

# Find cancellations for each booking channel
channel_cancellation = df.groupby("Booking_Channel").agg(
    Total_Bookings=("Booking_ID", "count"),
    Canceled_Bookings=("Canceled", lambda x: (x == "Yes").sum())
).reset_index()

channel_cancellation["Cancellation_Percentage"] = (
    channel_cancellation["Canceled_Bookings"]
    / channel_cancellation["Total_Bookings"] * 100
)

print("Cancellation by booking channel")
print(channel_cancellation.round(2))

# Make groups based on how early the booking was made
df["Lead_Time_Band"] = pd.cut(
    df["Lead_Time_Days"],
    bins=[-1, 7, 30, 90, float("inf")],
    labels=["Last-Minute", "Short-Term", "Medium-Term", "Advance"]
)

# Find cancellations in each lead-time group
lead_cancellation = df.groupby("Lead_Time_Band", observed=False).agg(
    Total_Bookings=("Booking_ID", "count"),
    Canceled_Bookings=("Canceled", lambda x: (x == "Yes").sum())
).reset_index()

lead_cancellation["Cancellation_Percentage"] = (
    lead_cancellation["Canceled_Bookings"]
    / lead_cancellation["Total_Bookings"] * 100
)

print("Cancellation by lead-time group")
print(lead_cancellation.round(2))

# Find the three months with the highest revenue
monthly_revenue = completed.groupby("Arrival_Month")[
    "Booking_Revenue"
].sum().sort_values(ascending=False).reset_index()

print("Top three revenue months")
print(monthly_revenue.head(3).round(2))

# Compare average stay of business and leisure travellers
stay_summary = completed.groupby("Traveler_Type")["Nights"].mean()

print("Average stay by traveller type")
print(stay_summary.round(2))

# Use NumPy to calculate room rate statistics
room_rates = completed["Room_Rate"].to_numpy()

print("Average room rate:", round(np.mean(room_rates), 2))
print("Middle room rate:", round(np.median(room_rates), 2))
print("Standard deviation:", round(np.std(room_rates), 2))

# Find unusually high booking values using the IQR method
booking_values = completed["Booking_Revenue"]

q1 = np.percentile(booking_values, 25)
q3 = np.percentile(booking_values, 75)

iqr = q3 - q1
upper_limit = q3 + (1.5 * iqr)

high_value_bookings = completed[
    completed["Booking_Revenue"] > upper_limit
]

print("Upper limit for high-value bookings:", round(upper_limit, 2))
print("High-value bookings")
print(high_value_bookings[
    ["Booking_ID", "Room_Type", "Nights", "Room_Rate", "Booking_Revenue"]
].head(10))

# Make a summary using room type and booking channel
room_channel_summary = completed.groupby(
    ["Room_Type", "Booking_Channel"]
).agg(
    Total_Bookings=("Booking_ID", "count"),
    Total_Room_Nights=("Nights", "sum"),
    Total_Revenue=("Booking_Revenue", "sum")
).reset_index()

print("Room type and booking channel summary")
print(room_channel_summary.round(2))

# Save the results in CSV files
room_summary.to_csv("room_type_revenue_summary.csv", index=False)
channel_summary.to_csv("channel_booking_revenue_summary.csv", index=False)
channel_cancellation.to_csv("channel_cancellation_summary.csv", index=False)
lead_cancellation.to_csv("lead_time_cancellation_summary.csv", index=False)
monthly_revenue.to_csv("monthly_revenue_summary.csv", index=False)
room_channel_summary.to_csv("room_type_channel_summary.csv", index=False)
high_value_bookings.to_csv("high_value_bookings.csv", index=False)

# Give five suggestions for hotel management
best_channel = channel_summary.iloc[0]["Booking_Channel"]

best_room = room_summary.sort_values(
    by="Total_Revenue", ascending=False
).iloc[0]["Room_Type"]

highest_cancellation_channel = channel_cancellation.sort_values(
    by="Cancellation_Percentage", ascending=False
).iloc[0]["Booking_Channel"]

longest_stay_type = stay_summary.idxmax()
highest_revenue_month = monthly_revenue.iloc[0]["Arrival_Month"]

print("Suggestions for hotel management")
print("1. Focus on the", best_channel, "booking channel because it has the most bookings.")
print("2. Review the pricing of", best_room, "rooms because they earn the most revenue.")
print("3. Check why", highest_cancellation_channel, "has the highest cancellation rate.")
print("4. Make suitable packages for", longest_stay_type, "travellers.")
print("5. Prepare for more customers in", highest_revenue_month, "because it has the highest revenue.")

print("Project analysis completed.")
