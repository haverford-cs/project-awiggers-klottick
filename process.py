import csv
source_file = "spreadspoke_scores.csv"
#dest_file

def main():
    read_csv(source_file)

def read_csv(source_file):
    final_list = []
    with open(source_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter =",")
        line_count = 1
        for row in csv_reader:
            if line_count >= 2503 and row[9] != '':
                datapoint = []
                #schedule season
                datapoint.append(int(row[1])-1979)
                #week
                datapoint = datapoint + weeks(row[2])
                #home team
                datapoint = datapoint + team_vals(row[3])
                #away team
                datapoint = datapoint + team_vals(row[4])
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

                #label
                label = 1
                if int(row[13]) > int(row[14]):
                    label = 2
                elif int(row[13]) < int(row[14]):
                    label = 0
                datapoint.append(label)
                #print(datapoint)
                final_list.append(datapoint)
            line_count +=1

        return final_list

def weeks(week):
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
    return team_features

if __name__ == "__main__" :
    main()
