from flask import Flask, render_template, request
import torch
import torch.nn as nn

input_size = 10
sequence_length = 3
num_classes = 1
num_layers = 1
hidden_size = 20
learning_rate = 0.00005


class RNN(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, num_classes):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(20, 5)
        self.fc2 = nn.Linear(5, 1)
    def forward(self, x):
        h0 = torch.rand(1, 20)
        # Forward Propogation
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        out = self.fc2(out)
        return out


model = RNN(input_size, num_layers, hidden_size, num_classes)
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

dataFile = 'data.txt'
all_data = torch.zeros([99, 40])
train = torch.zeros([75, 40])

coords_to_data = {}

with open(dataFile, 'r') as f:
    lines = [line for line in f.readlines()]
    elems = -1
    i = 0
    while i < len(lines):
        v = lines[i]
        long_lat = v
        long = float(v[4:13])
        lat = float(v[-9:])
        elems += 1
        while not lines[i].startswith('rasterColumn'):
            i += 1
        i += 1
        all_data[elems] = torch.zeros(40)
        pos = 0
        coords_to_data[(long, lat)] = []
        while i < len(lines) and lines[i].startswith('viirs'):
            start_pos = lines[i].find(',')
            coords_to_data[(long, lat)].append(float(lines[i][start_pos+1:]))
            all_data[elems][pos] = float(lines[i][start_pos + 1:])
            pos += 1
            i += 1
        while i < len(lines) and not lines[i].startswith('lon'):
            i += 1

# print(all_data)
train = all_data[:70]
test = all_data[71:]
criterion = nn.CrossEntropyLoss()

for sq in train:
    for i in range(30):
        sequence = sq[i:i+10]
        target = sq[i+10:i+11]
        target = target.view(1, 1)
        score = model(sequence.unsqueeze(0))
        loss = (target - score) * (target - score)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

for sq in test:
    with torch.no_grad():
        sequence = sq[29:39]
        target = sq[39:]
        target = target.view(1, 1)
        score = model(sequence.unsqueeze(0))
        # print(score, target)


def distance(target, coord):
  long1, lat1, long2, lat2 = target[0], target[1], coord[0], coord[1]
  return (long1 - long2) * (long1 - long2) + (lat1 - lat2) * (lat1 - lat2)


def predict_res(target, year):
    distances = []
    for coord in coords_to_data:
      distances.append(((distance(target, coord)), coord))
    distances.sort()
    ret = distances[:4]
    sum_hur = sum(coord[0] for coord in ret)
    time_series = torch.zeros(1, 40)
    for pos in range(40):
      for x in range(len(ret)):
        time_series[0][pos] += (ret[3-x][0] / sum_hur) * coords_to_data[ret[x][1]][pos]
    for yr in range(2022, year):
      # get last 10 years
      # print(time_series)
      # print(time_series[0][-10:].unsqueeze(0))
      # print(torch.cat((time_series, torch.tensor([[model(time_series[0][-10:].unsqueeze(0))]])), 1))
      time_series = torch.cat((time_series, torch.tensor([[model(time_series[0][-10:].unsqueeze(0))]])), 1)
      return time_series[0][-1]
    # print(coords_to_data)

def get_time_series(target):
    distances = []
    for coord in coords_to_data:
        distances.append(((distance(target, coord)), coord))
    distances.sort()
    ret = distances[:4]
    sum_hur = sum(coord[0] for coord in ret)
    time_series = [0 for _ in range(40)]
    for pos in range(40):
        for x in range(len(ret)):
            time_series[pos] += (ret[3 - x][0] / sum_hur) * coords_to_data[ret[x][1]][pos]
    return time_series

app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
    return render_template("index.html")

@app.route("/result", methods = ["POST", "GET"])
def result():
    output = request.form.to_dict()
    long = output["Longitude"]
    lat = output["Latitude"]
    year = output["Year"]
    res = int(predict_res((float(long), float(lat)), int(year)).item() * 1000) / 1000
    xArray = [yr for yr in range(2012, int(year)+1)]
    yArray = get_time_series((float(long), float(lat)))[::4] + [predict_res((float(long), float(lat)), yr).item() for yr in range(2023, int(year)+1)]

    return render_template("result.html", long = long, lat = lat, year = year, res = res, xArray = xArray, yArray = yArray)