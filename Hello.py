def run():
    import streamlit as st
    from streamlit.logger import get_logger

    LOGGER = get_logger(__name__)

    st.set_page_config(
        page_title="xaraf",
        page_icon="🚗🚙🚍",
    )

    st.write("# 🚗Predicting Car Sales with Machine Learning")
    st.write("Welcome to our journey into the exciting world of predicting car sales using advanced machine learning techniques! In this project, we delve into the realm of data-driven insights and predictive analytics to forecast the future sales of automobiles.")
    st.write("# 🚙Why Predict Car Sales?")
    st.write("Understanding and forecasting car sales is crucial for automakers, dealerships, and the entire automotive industry. Accurate predictions empower businesses to optimize inventory, marketing strategies, and overall decision-making. By leveraging the power of machine learning, we aim to unlock patterns and trends hidden within the data to make informed predictions about future car sales.")
    st.write("# 🚍Our Approach")
    st.write("We embark on this adventure armed with a dataset containing a wealth of information about various factors that influence car sales. From historical sales figures and marketing expenditures to socio-economic indicators and consumer preferences, our goal is to harness this data to build a robust predictive model. ")
    button_clicked = st.button("Let's Try Predict")
    
    # combined_app.py
    import pandas as pd
    import pickle
    #from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import GradientBoostingRegressor



    # Load X_train_n from the CSV file
    X_train_n = pd.read_csv('X_train.csv')

    # Load y_train_n from the CSV file
    y_train_n = pd.read_csv('y_train.csv')['target']

    # Now X_train_n and y_train_n should contain your data


    # Assuming X_train and y_train are your training data
    best_params = {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 100}

    # Instantiate the GradientBoostingRegressor with the best parameters
    final_model = GradientBoostingRegressor(learning_rate=best_params['learning_rate'],
                                            max_depth=best_params['max_depth'],
                                            n_estimators=best_params['n_estimators'],
                                            random_state=42)

    # Fit the model to the training data
    final_model.fit(X_train_n, y_train_n)


    # Define feature names to match the model's training
    feature_names = [
        'Price_in_thousands',
        'Manufacturer',
        'Length',
        '__year_resale_value',
        'Wheelbase',
    ]


    def get_user_inputs():
        user_inputs = {}
        for feature in feature_names:
            if feature == 'Manufacturer':
                # Provide a list of options for 'Manufacturer'
                manufacturer_options = ['Volvo', 'Audi', 'Acura', 'Chevrolet', 'Cadillac', 'Buick', 'BMW', 'Dodge', 'Toyota', 'Pontiac', 'Dodge', 'Chevrolet']
                selected_manufacturer = st.selectbox(f"Select {feature}:", options=manufacturer_options)
                # Encoding Manufacturer as 1, 2, 3, ...
                encoded_manufacturer = manufacturer_options.index(selected_manufacturer) + 1
                user_inputs[feature] = encoded_manufacturer
            else:
                # Use Streamlit widgets for numeric input
                value = st.number_input(f"Enter value for {feature}:")
                user_inputs[feature] = value

        return user_inputs



    # Use Streamlit form to collect all user inputs
    with st.form("user_inputs_form"):
        user_inputs = get_user_inputs()

        # Submit button
        submit_button = st.form_submit_button("Predict the sales")

    # After form submission, make the prediction
    if submit_button:
        # Create a DataFrame from user inputs
        user_input_df = pd.DataFrame([user_inputs])
        
        # Making prediction
        prediction = final_model.predict(user_input_df)
        
        # Convert prediction to int64
        y_pred_gbr = prediction.astype('int64')
        
        # Display the prediction
        # st.write("Predicted Value: ", y_pred_gbr)
        st.write("Predicted Value: ", f"<div style='text-align: center;'>{y_pred_gbr}</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    run()