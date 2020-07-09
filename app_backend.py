# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the model
filename = 'first-innings-score-lr-model.pkl'
regressor = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

venue_dicts = {
    'Barabati Stadium' : 0,
    'Brabourne Stadium' : 0, 
    'Buffalo Park' : 0, 
    'De Beers Diamond Oval' : 0, 
    'Dr DY Patil Sports Academy' : 0,
    'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium' : 0,
    'Dubai International Cricket Stadium' : 0,
    'Eden Gardens' : 0,
    'Feroz Shah Kotla' : 0,
    'Himachal Pradesh Cricket Association Stadium' : 0,
    'Holkar Cricket Stadium' : 0,
    'JSCA International Stadium Complex' : 0,
    'Kingsmead' : 0,
    'M Chinnaswamy Stadium' : 0,
    'MA Chidambaram Stadium, Chepauk' : 0,
    'Maharashtra Cricket Association Stadium' : 0,
    'New Wanderers Stadium' : 0,
    'Newlands' : 0,
    'OUTsurance Oval' : 0,
    'Punjab Cricket Association IS Bindra Stadium, Mohali' : 0,
    'Rajiv Gandhi International Stadium, Uppal' : 0,
    'Sardar Patel Stadium, Motera' : 0,
    'Sawai Mansingh Stadium' : 0,
    'Shaheed Veer Narayan Singh International Stadium' : 0,
    'Sharjah Cricket Stadium' : 0,
    'Sheikh Zayed Stadium' : 0,
    "St George's Park" : 0, 
    'Subrata Roy Sahara Stadium' : 0,
    'SuperSport Park' : 0,
    'Wankhede Stadium' : 0}
    
# bat_team_dict = {'Chennai Super Kings' : 0, 'Delhi Daredevils' : 0, 'Kings XI Punjab' : 0, 'Kolkata Knight Riders' : 0, 'Mumbai Indians' : 0, 'Rajasthan Royals' : 0, 'Royal Challengers Bangalore' : 0, 'Sunrisers Hyderabad' : 0}
       
# bowl_team_dict = {'Chennai Super Kings' : 0, 'Delhi Daredevils' : 0,
   # 'Kings XI Punjab' : 0, 'Kolkata Knight Riders' : 0,
   # 'Mumbai Indians' : 0, 'Rajasthan Royals' : 0,
   # 'Royal Challengers Bangalore' : 0, 'Sunrisers Hyderabad' : 0}

@app.route('/')
def home():
    return render_template('index.html', venue_list = list(venue_dicts.keys()))

@app.route('/predict', methods=['POST'])
def predict():
    temp_array = list()
    
    venue_dict = {
    'Barabati Stadium' : 0,
    'Brabourne Stadium' : 0, 
    'Buffalo Park' : 0, 
    'De Beers Diamond Oval' : 0, 
    'Dr DY Patil Sports Academy' : 0,
    'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium' : 0,
    'Dubai International Cricket Stadium' : 0,
    'Eden Gardens' : 0,
    'Feroz Shah Kotla' : 0,
    'Himachal Pradesh Cricket Association Stadium' : 0,
    'Holkar Cricket Stadium' : 0,
    'JSCA International Stadium Complex' : 0,
    'Kingsmead' : 0,
    'M Chinnaswamy Stadium' : 0,
    'MA Chidambaram Stadium, Chepauk' : 0,
    'Maharashtra Cricket Association Stadium' : 0,
    'New Wanderers Stadium' : 0,
    'Newlands' : 0,
    'OUTsurance Oval' : 0,
    'Punjab Cricket Association IS Bindra Stadium, Mohali' : 0,
    'Rajiv Gandhi International Stadium, Uppal' : 0,
    'Sardar Patel Stadium, Motera' : 0,
    'Sawai Mansingh Stadium' : 0,
    'Shaheed Veer Narayan Singh International Stadium' : 0,
    'Sharjah Cricket Stadium' : 0,
    'Sheikh Zayed Stadium' : 0,
    "St George's Park" : 0, 
    'Subrata Roy Sahara Stadium' : 0,
    'SuperSport Park' : 0,
    'Wankhede Stadium' : 0}
    
    bat_team_dict = {'Chennai Super Kings' : 0, 'Delhi Daredevils' : 0, 'Kings XI Punjab' : 0, 'Kolkata Knight Riders' : 0, 'Mumbai Indians' : 0, 'Rajasthan Royals' : 0, 'Royal Challengers Bangalore' : 0, 'Sunrisers Hyderabad' : 0}
       
    bowl_team_dict = {'Chennai Super Kings' : 0, 'Delhi Daredevils' : 0,
       'Kings XI Punjab' : 0, 'Kolkata Knight Riders' : 0,
       'Mumbai Indians' : 0, 'Rajasthan Royals' : 0,
       'Royal Challengers Bangalore' : 0, 'Sunrisers Hyderabad' : 0}
    
    if request.method == 'POST':
    
        
        batting_team = request.form['batting-team']
        bat_team_dict[batting_team] = 1
        temp_array += list(bat_team_dict.values())
            
            
        bowling_team = request.form['bowling-team']
        bowl_team_dict[bowling_team] = 1
        temp_array += list(bowl_team_dict.values())
            
        
        venue = request.form['venue']
        venue_dict[venue] = 1
        temp_array += list(venue_dict.values())
        
        
        runs = int(request.form['runs'])
        wickets = int(request.form['wickets'])
        overs = float(request.form['overs'])
        runs_in_prev_5 = int(request.form['runs_in_prev_5'])
        wickets_in_prev_5 = int(request.form['wickets_in_prev_5'])
        
        temp_array = temp_array + [runs, wickets, overs, runs_in_prev_5, wickets_in_prev_5]
        
        
        
        
        data = np.array([temp_array])
        my_prediction = int(regressor.predict(data)[0])
              
        return render_template('result.html', lower_limit = my_prediction-5, upper_limit = my_prediction+5, temp = temp_array, d = data, p = my_prediction)



if __name__ == '__main__':
    app.run(debug=True)