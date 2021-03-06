"""
Import and parse football data.
Author: Alton Wiggers, Kadan Lottick
Date: 12/17/19
"""

import csv
source_file = "spreadspoke_scores.csv"
#dest_file

def main():
    read_csv(source_file,1979)

def get_bookie_score(source_file):
    """
    check predicted winners against actual results
    for all games in 2015-2017 seasons
    """
    with open(source_file) as csv_file:
        total = 0
        correct = 0
        csv_reader = csv.reader(csv_file, delimiter =",")
        line_count = 1
        for row in csv_reader:
            if line_count >= 2503 and int(row[1]) >= 2015 and int(row[1]) < 2019:
                if int(row[13]) > int(row[14]):
                    if row[3] == row[6]:
                        correct+=1
                elif int(row[13]) < int(row[14]):
                    if (row[3] != row[6] and row[6] != "PICK"):
                        correct+=1
                else:
                    if (row[6] == "PICK"):
                        correct+=1
                total+=1
            line_count+=1
        return correct, total, correct /total

def read_csv(source_file, min_year):
    """
    Parse in each line of .csv file as a datapoint.
    Convert relevant lines to analyzable features.
    Divide points into training and testing data.
    Discard datapoints from seasons before given min_year.
    """
    train_data = []
    test_data = []
    current_data = []

    #array indexed by years (subtract of 1979 to get index),
    #teams (assigned indexes by team_vals), wins/losses/ties (0/1/2)
    records = []
    for i in range(41):
        year = []
        records.append(year)
        for j in range(32):
            team = []
            records[i].append(team)
            for k in range(3):
                records[i][j].append(0)

    #array indexed by years (subtract of 1979 to get index),
    #teams (assigned indexes by team_vals), teams they've played
    #the value is the number of times the 1st team has beaten the 2nd that season
    winMatrix = []
    for i in range(41):
        year = []
        winMatrix.append(year)
        for j in range(32):
            team = []
            winMatrix[i].append(team)
            for k in range(32):
                winMatrix[i][j].append(0)


    with open(source_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter =",")
        line_count = 1
        #creating one datapoint for each game
        for row in csv_reader:
            #only using games after odds were given
            if line_count >= 2503 and row[8] != ' ' and row[8] != '' and \
                 row[10] != ' ' and row[10] != '':
                datapoint = []
                #schedule season
                year = int(row[1])-1979
                datapoint.append(year)
                #week
                datapoint = datapoint + weeks(row[2])
                #home team
                home_vals,home_id = team_vals(row[3])
                datapoint = datapoint + home_vals
                #away team
                away_vals,away_id = team_vals(row[4])
                datapoint = datapoint + away_vals
                #spread
                if row[3] == row[6]:
                    datapoint.append(float(row[7]))
                else:
                    datapoint.append(-1*float(row[7]))
                #overUnder
                datapoint.append(float(row[8]))
                #temperature
                datapoint.append(float(row[10]))
                #wind
                datapoint.append(float(row[11]))
                #stadium neutral
                if row[15] == "FALSE":
                    datapoint.append(0)
                else:
                    datapoint.append(1)

                #add records for home and away teams this season
                if not year +1979 == 2019:
                    datapoint.append(records[year][home_id][0])
                    datapoint.append(records[year][home_id][1])
                    datapoint.append(records[year][home_id][2])
                    datapoint.append(records[year][away_id][0])
                    datapoint.append(records[year][away_id][1])
                    datapoint.append(records[year][away_id][2])
                else:
                    datapoint.append(float(row[17]))
                    datapoint.append(float(row[18]))
                    datapoint.append(float(row[19]))
                    datapoint.append(float(row[20]))
                    datapoint.append(float(row[21]))
                    datapoint.append(float(row[22]))

                #append matrix of whose beaten who
                for i in range(32):
                    for j in range(32):
                        datapoint.append(winMatrix[year][i][j])

                #assign label and update records
                label = 0
                #home wins
                if int(row[13]) > int(row[14]):
                    label = 0
                    records[year][home_id][0] +=1
                    records[year][away_id][1] +=1
                    winMatrix[year][home_id][away_id] +=1
                #away wins
                elif int(row[13]) < int(row[14]):
                    label = 1
                    records[year][home_id][1] +=1
                    records[year][away_id][0] +=1
                    winMatrix[year][away_id][home_id] +=1
                #tie
                else:
                    label = 2
                    records[year][home_id][2] +=1
                    records[year][away_id][2] +=1

                datapoint.append(label)
                if year + 1979 >= min_year and label < 2:
                    if year + 1979 < 2015:
                        train_data.append(datapoint)
                    elif year + 1979 < 2019:
                        test_data.append(datapoint)
                    else:
                        current_data.append(datapoint)
            line_count +=1

        return train_data,test_data,current_data

def weeks(week):
    """
    Given a week, create a list of one-hot
    features for each possible week and
    enable only the given value
    """

    list = []
    switcher = {
        "1": 0,
        "2": 1,
        "3": 2,
        "4": 3,
        "5": 4,
        "6": 5,
        "7": 6,
        "8": 7,
        "9": 8,
        "10": 9,
        "11": 10,
        "12": 11,
        "13": 12,
        "14": 13,
        "15": 14,
        "16": 15,
        "17": 16,
        "18": 17,
        "Wildcard": 18,
        "WildCard": 18,
        "Division": 19,
        "Conference": 20,
        "Superbowl": 21,
        "SuperBowl": 21
    }
    for i in range(22):
        list.append(0)
    list[switcher.get(week, "error")] = 1
    return list

def team_vals(team_id):
    """
    Given a team, create a list of one-hot
    features for each possible team and
    enable only the given value.
    Return the team value for generating
    other features.
    """

    switch = {
        "NE":0,
        "BUF":1,
        "NYJ":2,
        "MIA":3,
        "KC":4,
        "OAK":5,
        "DEN":6,
        "LAC":7,
        "BAL":8,
        "PIT":9,
        "CLE":10,
        "CIN":11,
        "HOU":12,
        "TEN":13,
        "IND":14,
        "JAX":15,
        "DAL":16,
        "PHI":17,
        "WAS":18,
        "NYG":19,
        "SEA":20,
        "SF":21,
        "LAR":22,
        "ARI":23,
        "GB":24,
        "MIN":25,
        "CHI":26,
        "DET":27,
        "NO":28,
        "TB":29,
        "CAR":30,
        "ATL":31
    }

    pos_id = switch.get(team_id, "No Team")
    team_features = []

    for i in range(32):
        team_features.append(0)

    team_features[pos_id] = 1
    return team_features,pos_id

if __name__ == "__main__" :
    main()
